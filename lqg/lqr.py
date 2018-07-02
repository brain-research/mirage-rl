#!python

import pickle
import sys
import numpy as np
from plotter import plot_trajs, plot_variances

def is_sym(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def logdet(X, assert_pd=False):
    sign, logdet = np.linalg.slogdet(X)
    if assert_pd:
        assert sign > 0, '%s,logdet=%f,sign=%s'%(X,logdet,sign)
    return sign, logdet

def logsumdiag(X, assert_pd=False):
    diag = np.diag(X)
    sign = np.all(diag > 0)
    diag = np.absolute(diag)
    sumdiag = np.sum(diag)
    if assert_pd:
        assert sumdiag > 0, '%s,logsumdiag=%f'%(X,sumdiag)
    return sign, np.log(sumdiag)

def tile_arr(X, mshape, ndim):
    mshape = tuple(mshape)
    _message = '%s, %s, %s' % (X.shape, ndim, mshape)
    if X.ndim == len(mshape):
        X = np.array(X)
        tiled = False
    elif X.ndim == ndim:
        X = np.tile(X[(None,)*(len(mshape)-ndim)], mshape[:-ndim] + (1,)*ndim)
        tiled = True
    else: raise NotImplementedError(_message)
    assert X.shape == mshape, _message
    return X, tiled

class GlobalLQG(object):

    def __init__(self, gamma, T, sdim, adim, A, B, Sigma_s_sa, Q, R):
        self.gamma = gamma
        self.T = T
        self.sdim = sdim
        self.adim = adim
        self.A, self.A_tiled = tile_arr(A, (T-1,sdim,sdim), 2)
        self.B, self.B_tiled = tile_arr(B, (T-1,sdim,adim), 2)
        self.Sigma_s_sa, self.Sigma_s_sa_tiled = tile_arr(Sigma_s_sa, (T,sdim,sdim), 2)
        self.Q, self.Q_tiled = tile_arr(Q, (T,sdim,sdim), 2)
        self.R, self.R_tiled = tile_arr(R, (T,adim,adim), 2)
        self._pi_set = False
        self.running = dict()

    def _batch_quad_fn(self, coefs, t, s, a=None):
        assert s.ndim == 2
        assert s.shape[1] == self.sdim
        res_ss = np.sum(np.dot(s, coefs['P_ss'][t]) * s, axis=1)
        res_s = np.dot(s, coefs['p_s'][t])
        res = res_ss + res_s + coefs['c'][t]
        if a is not None:
            assert a.ndim == 2
            assert a.shape[1] == self.adim
            assert s.shape[0] == a.shape[0]
            res_aa = np.sum(np.dot(a, coefs['P_aa'][t]) * a, axis=1)
            res_sa = np.sum(np.dot(s, coefs['P_sa'][t]) * a, axis=1)
            res_a = np.dot(a, coefs['p_a'][t])
            res +=  res_sa + res_aa + res_a
            assert not np.isnan(res_aa).any()
            assert not np.isnan(res_sa).any()
            assert not np.isnan(res_a).any()
        res *= float(coefs['sign'])
        assert not np.isnan(res_ss).any()
        assert not np.isnan(res_s).any()
        assert not np.isnan(res).any()
        return res

    def batch_q_fn(self, t, s, a):
        return self._batch_quad_fn(self.Qcoefs, t, s, a)

    def batch_v_fn(self, t, s):
        return self._batch_quad_fn(self.Vcoefs, t, s)

    def batch_a_fn(self, t, s, a):
        return self._batch_quad_fn(self.Acoefs, t, s, a)

    def batch_loggrad_fn(self, t, a):
        assert a.ndim == 2
        assert a.shape[1] == self.adim
        mu_grad = np.dot(a - self.mu_a[t][None], self.Sigmainv_a[t])
        Sigma_grad = self.Sigmainv_a[t][None] - mu_grad[:, None] * mu_grad[:, :, None]
        Sigma_grad *= -0.5
        assert not np.isnan(mu_grad).any()
        assert not np.isnan(Sigma_grad).any()
        return mu_grad, Sigma_grad

    def batch_E_grad_mu_fn(self, t, s):
        grad = np.dot(s, self.P_sa[t]) + np.dot(self.mu_a[t][None], self.P_aa[t])
        grad *= -1.0
        assert not np.isnan(grad).any()
        return grad

    def E_grad_mu_brute_fn(self, t, n):
        bs, ba, _, r = self.rollout(t, n)
        a = ba[0]
        mu_grad, _ = self.batch_loggrad_fn(t, a)
        r_mean = np.mean(r)
        grad = np.mean(mu_grad * (r-r_mean)[:, None], axis=0)
        assert not np.isnan(grad).any()
        return grad

    def E_grad_mu_brute2_fn(self, t, n):
        s = np.random.multivariate_normal(self.mu_s[t], self.Sigma_s[t], n)
        a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], n)
        mu_grad, _ = self.batch_loggrad_fn(t, a)
        Q = self.batch_q_fn(t, s, a)
        A = self.batch_a_fn(t, s, a)
        A2 = np.sum(np.dot(s, self.P_sa[t]) * a, axis=1)
        A2 += np.sum(np.dot(a, self.P_aa[t]) * a, axis=1)
        A2 += np.dot(a, self.p_a[t])
        A2 *= -1.0
        grad1 = np.mean(mu_grad * Q[:, None], axis=0)
        grad2 = np.mean(mu_grad * A[:, None], axis=0)
        grad3 = np.mean(mu_grad * A2[:, None], axis=0)
        return grad1, grad2, grad3

    def E_grad_mu_fn(self, t):
        grad = np.dot(self.mu_s[t], self.P_sa[t]) + 2.0*np.dot(self.mu_a[t], self.P_aa[t])
        grad += self.p_a[t]
        grad *= -1.0
        assert not np.isnan(grad).any()
        return grad

    def E_grad_L_fn(self, t, n):
        a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], n)
        _, Sigma_grad = self.batch_loggrad_fn(t, a)
        signal = np.dot(a, np.dot(self.mu_s[t], self.P_sa[t]))
        signal += np.sum(np.dot(a, self.P_aa[t]) * a, axis=1)
        signal *= -1.0
        Sigma_grad *= signal[:, None, None]
        Sigma_grad = np.mean(Sigma_grad, axis=0)
        phi_in = np.dot(np.dot(self.Linv_a[t], Sigma_grad), self.Linv_a[t].T)
        phi = np.tril(phi_in) - np.diag(np.diag(phi_in)) * 0.5
        grad = np.dot(self.L_a[t], phi)
        assert ((np.tril(grad.T) - np.diag(np.diag(grad))) == 0).all(), \
                '%s,%s,%s' % (grad.T, np.tril(grad.T), np.tril(grad.T) == 0)
        assert not np.isnan(grad).any()
        return grad

    def test_value_fns(self, t, n=10, eps=1e-7):
        s = np.random.multivariate_normal(self.mu_s[t], self.Sigma_s[t], n)
        a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], n)
        Q = self.batch_q_fn(t, s, a)
        A = self.batch_a_fn(t, s, a)
        V = self.batch_v_fn(t, s)
        V0 = Q - A
        assert np.allclose(V0, V, atol=eps), '%s,%s,%s,%s' % (Q,A,V,V0)
        print(Q,A,V,V0)

    def test_rollout(self, t, n=1000,):
        s = np.random.multivariate_normal(self.mu_s[t], self.Sigma_s[t], 1)
        a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], 1)
        q = self.batch_q_fn(t, s, a)[0]
        s = np.tile(s, (n, 1))
        a = np.tile(a, (n, 1))
        bs, ba, br, r, gae_rs = self.rollout(t, n, s=s, a=a, lambdas=[0.0, 0.1, 0.9])
        r = np.mean(r)
        gae_r = [np.mean(gae_r_) for gae_r_ in gae_rs]
        print(q, r, gae_r)

    def test_grad_mu(self, t, eps=1e-7):
        grad = self.E_grad_mu_fn(t)
        grad1, grad2, grad3 = self.E_grad_mu_brute2_fn(t, 1000000)
        grad12, grad22, grad32 = self.E_grad_mu_brute2_fn(t, 1000000)
        fin_grad = np.zeros_like(grad)
        mu = np.array(self.mu_a[t])
        for i in range(self.adim):
            self.mu_a[t, i] += eps
            self.precompute_pi_conditional() # update pi conditional matrices
            rp, _ = self.J(t)
            self.mu_a[t, i] -= 2.0*eps
            self.precompute_pi_conditional() # update pi conditional matrices
            rm, _ = self.J(t)
            fin_grad[i] = (rp-rm)/(2.0*eps)
            self.mu_a[t][:] = mu
        self.precompute_pi_conditional() # update pi conditional matrices
        print(t, grad, grad1, grad2, grad3, grad12, grad22, grad32, fin_grad)

    def J(self, t):
        r = 0
        undiscounted_r = 0
        for i in range(t, self.T):
            r_t = self.E_r_fn(t)
            r += self.gamma**(i-t) * r_t
            undiscounted_r += r_t
        return r, undiscounted_r

    def E_r_fn(self, t):
        r = np.dot(np.dot(self.mu_s[t], self.Q[t]), self.mu_s[t])
        r += np.trace(np.dot(self.Q[t], self.Sigma_s[t]))
        r += np.dot(np.dot(self.mu_a[t], self.R[t]), self.mu_a[t])
        r += np.trace(np.dot(self.R[t], self.Sigma_a[t]))
        r *= -1.0
        return r

    def optimize(self, opt='sgd', ignore_Sigma=True, opt_params=None, n=5000):
        mu_grads = np.zeros_like(self.mu_a)
        L_grads = np.zeros_like(self.L_a)
        rs = np.zeros((self.T, ))
        for t in range(self.T):
            mu_grads[t] = self.E_grad_mu_fn(t)
            #mu_grads[t] = self.E_grad_mu_brute_fn(t, n)
            L_grads[t] = self.E_grad_L_fn(t, n)
            rs[t] = self.E_r_fn(t)
            if 'ng' in opt_params and opt_params['ng']:
                mu_grads[t] /= np.diag(self.Sigma_a[t])

        if opt == 'sgd':
            lr = opt_params['lr']
            mom = opt_params['mom']
            if 'mu_grads' not in self.running:
                self.running['mu_grads'] = np.array(mu_grads)
                self.running['L_grads'] = np.array(L_grads)
            else:
                self.running['mu_grads'] = mom * self.running['mu_grads']+ (1.0-mom) * mu_grads
                self.running['L_grads'] = mom * self.running['L_grads'] + (1.0-mom) * L_grads
            self.mu_a += lr * self.running['mu_grads']
            if not ignore_Sigma:
                self.L_a += lr * self.running['L_grads']
        else:
            NotImplementedError(opt)
        self.precompute_pi_conditional() # update pi conditional matrices
        return rs.sum()

    def est_sigma_s(self, t):
        sigma = np.dot(np.dot(self.P_sa[t].T, self.Sigma_s[t]), self.P_sa[t])
        assert not np.isnan(sigma).any()
        return sigma

    def est_sigma_a(self, t, n):
        s = np.random.multivariate_normal(self.mu_s[t], self.Sigma_s[t], n)
        a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], n)
        Q = self.batch_q_fn(t, s, a)
        A = self.batch_a_fn(t, s, a)
        loggrad_mu, _ = self.batch_loggrad_fn(t, a)
        E_grad_mu = self.batch_E_grad_mu_fn(t, s)
        loggrad_mu_cross = loggrad_mu[:, None] * loggrad_mu[:, :, None]
        E_grad_mu_cross = E_grad_mu[:, None] * E_grad_mu[:, :, None]
        Q2 = Q**2
        A2 = A**2
        sigma_Q = np.mean(Q2[:, None, None] * loggrad_mu_cross - E_grad_mu_cross, axis=0)
        sigma_A = np.mean(A2[:, None, None] * loggrad_mu_cross - E_grad_mu_cross, axis=0)
        sigma_V = np.mean((Q2 - A2)[:, None, None] * loggrad_mu_cross, axis=0)
        assert not np.isnan(sigma_Q).any()
        assert not np.isnan(sigma_A).any()
        assert not np.isnan(sigma_V).any()
        return sigma_Q, sigma_A, sigma_V

    def est_sigma_tau(self, t, n, lambdas):
        bs, ba, _, r, gae_rs = self.rollout(t, n, lambdas=lambdas)
        s = bs[0]
        a = ba[0]
        Q = self.batch_q_fn(t, s, a)
        Q2 = Q**2
        loggrad_mu, _ = self.batch_loggrad_fn(t, a)
        loggrad_mu_cross = loggrad_mu[:, None] * loggrad_mu[:, :, None]
        sigma = np.mean(loggrad_mu_cross * ((r **2 - Q2) [:, None, None]), axis=0)
        sigma_lambdas = []
        for gae_r in gae_rs:
            sigma_lambdas.append(np.mean(loggrad_mu_cross * ((gae_r **2 - Q2)
                [:, None, None]), axis=0))

        assert not np.isnan(sigma).any()
        for sigma_lambda in sigma_lambdas:
            assert not np.isnan(sigma_lambda).any()
        return sigma, sigma_lambdas

    def get_variances(self, n, lambdas=[0.99], variance_fn='logdet'):
        if variance_fn == 'logdet':
            variance_fn = logdet
        elif variance_fn == 'logsumdiag':
            variance_fn = logsumdiag
        else:
            raise NotImplementedError(variance_fn)
        sigma_tau = np.zeros(self.T-1)
        sigma_tau_ls = [np.zeros(self.T-1) for _ in lambdas]
        sigma_tau = np.zeros(self.T-1)
        sigma_s = np.zeros(self.T-1)
        sigma_a_Q = np.zeros(self.T-1)
        sigma_a_A = np.zeros(self.T-1)
        sigma_a_V = np.zeros(self.T-1)
        for t in range(self.T-1):
            est_tau, est_tau_lambdas = self.est_sigma_tau(t, n, lambdas)
            est_s = self.est_sigma_s(t)
            est_a_Q, est_a_A, est_a_V = self.est_sigma_a(t, n)
            sign_tau ,sigma_tau[t] = variance_fn(est_tau)
            for i, est_tau_l in enumerate(est_tau_lambdas):
                sign_tau_l ,sigma_tau_ls[i][t] = variance_fn(est_tau_l)
                if sign_tau_l <= 0:
                    print('WARNING: t=%d,l=%f,%s,variance_fn=%f,sign_tau_l=%s'%(
                        t,lambdas[i],est_tau_l,sigma_tau_ls[i][t],sign_tau_l))
            sign_s, sigma_s[t] = variance_fn(est_s)
            sign_a_Q, sigma_a_Q[t] = variance_fn(est_a_Q)
            sign_a_A, sigma_a_A[t] = variance_fn(est_a_A)
            sign_a_V, sigma_a_V[t] = variance_fn(est_a_V)
            if sign_tau <= 0:
                print('WARNING: t=%d,%s,variance_fn=%f,sign_tau=%s'%(t,est_tau,sigma_tau[t],sign_tau))
        variances = {
                r'$\Sigma_\tau$': sigma_tau,
                r'$\Sigma_s$': sigma_s,
                #r'$\Sigma_a^Q$': sigma_a_Q,
                #r'$\Sigma_a^A$': sigma_a_A,
                r'$\Sigma_a^0$': sigma_a_Q,
                r'$\Sigma_a^{\phi(s)}$': sigma_a_A,
                #r'$\Sigma_a^{Q-A}$': sigma_a_V,
                }
        for lambda_, sigma_tau_l in zip(lambdas, sigma_tau_ls):
            variances[r'$\Sigma_\tau^{GAE(\lambda=%s)}$'%lambda_] = sigma_tau_l
            #variances[r'$\Sigma_\tau^{GAE-%s}$'%lambda_] = sigma_tau_l
        return variances

    def rollout(self, t, n, s=None, a=None, lambdas=[], gae_opt='inf'):
        bs = np.zeros((self.T-t, n, self.sdim))
        ba = np.zeros((self.T-t, n, self.adim))
        br = np.zeros((self.T-t, n, ))
        r = np.zeros((n,))
        gae_rs = [np.zeros((n,)) for _ in lambdas]
        if a is None:
            a = np.random.multivariate_normal(self.mu_a[t], self.Sigma_a[t], n)
        for i in range(t, self.T):
            if i == t:
                if s is None:
                    bs[i-t] = np.random.multivariate_normal(self.mu_s[t], self.Sigma_s[t], n)
                else:
                    bs[i-t] = s
            else:
                mu_s_sa = np.dot(bs[i-t-1], self.A[i-1].T) + np.dot(ba[i-t-1], self.B[i-1].T)
                bs[i-t] = np.random.multivariate_normal(np.zeros((self.sdim,)), self.Sigma_s[i], n)
                bs[i-t] += mu_s_sa
            if i == t and a is not None:
                ba[i-t] = a
            else:
                ba[i-t] = np.random.multivariate_normal(self.mu_a[i], self.Sigma_a[i], n)
            br[i-t] = np.sum(np.dot(bs[i-t], self.Q[i]) * bs[i-t], axis=1)
            br[i-t] += np.sum(np.dot(ba[i-t], self.R[i]) * ba[i-t], axis=1)
            br[i-t] *= -1.0
            r += self.gamma ** (i-t) * br[i-t]
            for lambda_, gae_r in zip(lambdas, gae_rs):
                if gae_opt == 'fin':
                    # horizon corrected GAE
                    norm = 1.0 - lambda_ ** (self.T-t-1)
                    if i > t:
                        v = self.batch_v_fn(i, bs[i-t])
                        gae_r += (((self.gamma * lambda_) ** (i-t-1)) * self.gamma * (1.0-lambda_)/ norm) * v
                        #print(i, r[0], gae_r[0], v[0], norm)
                    if i < self.T-1:
                        coef1 = (self.gamma * lambda_) ** (i-t)
                        coef2 = (1.0 - lambda_ ** (self.T-i-1))
                        temp1 = coef1 * coef2 * br[i-t] / norm
                        gae_r += temp1
                        #print(i, r[0], gae_r[0], coef1, coef2, norm, temp1[0], br[i-t][0])
                elif gae_opt == 'inf':
                    # fixed GAE
                    gamma_ = self.gamma * lambda_
                    if i == t:
                        gae_r += br[i-t]
                    else:
                        gae_r +=  (gamma_ ** (i-t)) * br[i-t]
                        v = self.batch_v_fn(i, bs[i-t])
                        gae_r += (-gamma_ ** (i-t) + self.gamma * gamma_ ** (i-t-1)) * v
        return bs, ba, br, r, gae_rs

    def set_pi(self, mu_a, L_a, mu_s_0):
        assert mu_a.shape == (self.T, self.adim)
        assert L_a.shape == (self.T, self.adim, self.adim)
        assert mu_s_0.shape == (self.sdim,)
        self.mu_a = np.array(mu_a)
        self.L_a = np.array(L_a)
        self.mu_s_0 = np.array(mu_s_0)
        self._pi_set = True

    def get_pi(self):
        return dict(
                mu_a=self.mu_a,
                L_a=self.L_a,
                mu_s_0=self.mu_s_0,
                )

    def compute_Sigma(self):
        self.Sigma_a = np.zeros_like(self.L_a)
        self.Linv_a = np.zeros_like(self.L_a)
        self.Sigmainv_a = np.zeros_like(self.L_a)
        for t in range(self.T):
            self.Sigma_a[t] = np.dot(self.L_a[t], self.L_a[t].T)
            self.Linv_a[t] = np.linalg.inv(self.L_a[t])
            self.Sigmainv_a[t] = np.linalg.inv(self.Sigma_a[t])
        assert not np.isnan(self.Linv_a).any()
        assert not np.isnan(self.Sigmainv_a).any()

    def precompute(self):
        self.compute_L()
        self.compute_P()

    def precompute_pi_conditional(self):
        self.compute_Sigma()
        self.compute_mM()
        self.compute_pc()
        self.compute_qa_coefs()
        self.compute_mu_Sigma_s()

    def compute_L(self):
        assert self.A_tiled and self.B_tiled
        self._L = np.zeros((self.T+1, self.sdim, self.sdim))
        self._LB = np.zeros((self.T, self.sdim, self.adim))
        self._L.fill(np.nan)
        self._LB.fill(np.nan)
        self._L[0].fill(0) # index 0 is zeros/invalid
        self._L[1] = np.eye(self.sdim) # index 1 is identity
        for i in range(2, self.T+1):
            self._L[i] = np.dot(self.A[0], self._L[i-1])
        self.L = np.tile(self._L[None], (self.T, 1, 1, 1))
        for i in range(self.T):
            self._LB[i] = np.dot(self._L[i], self.B[0])
        self.LB = np.tile(self._LB[None], (self.T, 1, 1, 1))
        assert not np.isnan(self.L).any()
        assert not np.isnan(self.LB).any()

    def compute_mM(self):
        self.m = np.zeros((self.T, self.T, self.sdim))
        self.m.fill(np.nan)
        self.m[:, 0].fill(0) # second index 0 is zeros
        self.M = np.zeros((self.T, self.T, self.sdim, self.sdim))
        self.M.fill(np.nan)
        self.M[:, 0].fill(0) # second index 0 is zeros
        for i in range(1, self.T):
            for j in range(0, i):
                m_km1 = self.m[j, i-j-1]
                M_km1 = self.M[j, i-j-1]
                assert not np.isnan(m_km1).any(), \
                        '%s,i=%d,j=%d'%(np.isnan(self.m).any(axis=2), i, j)
                assert not np.isnan(M_km1).any(), \
                        '%s,i=%d,j=%d'%(np.isnan(self.M).any(axis=(2,3)), i, j)
                self.m[j, i-j] = np.dot(self.A[i-1], m_km1
                        ) + np.dot(self.B[i-1], self.mu_a[i-1])
                self.M[j, i-j] = np.dot(np.dot(self.A[i-1], M_km1), self.A[i-1].T
                        ) + np.dot(np.dot(self.B[i-1], self.Sigma_a[i-1]), self.B[i-1].T
                        ) + self.Sigma_s_sa[i]


    def compute_mu_Sigma_s(self):
        self.mu_s = np.zeros((self.T, self.sdim))
        self.Sigma_s = np.zeros((self.T, self.sdim, self.sdim))
        self.mu_s.fill(np.nan)
        self.Sigma_s.fill(np.nan)
        self.mu_s[0] = np.array(self.mu_s_0)
        self.Sigma_s[0] = np.array(self.Sigma_s_sa[0])
        for t in range(1, self.T):
            self.mu_s[t] = np.dot(self.A[t-1], self.mu_s[t-1]) + np.dot(self.B[t-1], self.mu_a[t-1])
            self.Sigma_s[t] = np.dot(np.dot(self.A[t-1], self.Sigma_s[t-1]), self.A[t-1].T
                    ) + np.dot(np.dot(self.B[t-1], self.Sigma_a[t-1]), self.B[t-1].T
                    ) + self.Sigma_s_sa[t]
            assert is_sym(self.Sigma_s[t]), self.Sigma_s[t]
        assert not np.isnan(self.mu_s).any()
        assert not np.isnan(self.Sigma_s).any()

    def compute_pc(self):
        assert self.A_tiled and self.Q_tiled and self.R_tiled
        Q = self.Q[0]
        R = self.R[0]
        self.p_s = np.zeros((self.T, self.sdim))
        self.p_a = np.zeros((self.T, self.adim))
        self.c = np.zeros((self.T,))
        for t in range(self.T-1):
            for k in range(1, self.T-t):
                m = self.m[t+1, k-1]
                M = self.M[t+1, k-1]
                assert not np.isnan(m).any(), \
                        '%s,t=%d,k=%d'%(np.isnan(self.m).any(axis=2),t,k)
                assert not np.isnan(M).any(), np.isnan(self.M).any(axis=(2,3,))
                self.p_s[t] += (self.gamma ** k) * np.dot(np.dot(
                    self.L[0, k+1].T, Q), m)
                self.p_a[t] += (self.gamma ** k) * np.dot(np.dot(
                    self.LB[0, k].T, Q), m)
                c_t = np.dot(np.dot(m, Q), m)
                c_t += np.dot(np.dot(self.mu_a[t+k], R), self.mu_a[t+k])
                c_t += np.trace(np.dot(R, self.Sigma_a[t+k]))
                Sigma_t_k_s = np.dot(np.dot(self.L[0, k].T, self.Sigma_s_sa[t+k]),
                        self.L[0, k]) + M
                c_t += np.trace(np.dot(Q, Sigma_t_k_s))
                c_t *= self.gamma ** k
                self.c[t] += c_t
        self.p_s *= 2
        self.p_a *= 2
        assert not np.isnan(self.p_s).any()
        assert not np.isnan(self.p_a).any()
        assert not np.isnan(self.c).any()

    def compute_qa_coefs(self):
        self.Qcoefs = dict(
                P_ss = self.P_ss,
                P_sa = self.P_sa,
                P_aa = self.P_aa,
                p_s = self.p_s,
                p_a = self.p_a,
                c = self.c,
                sign = -1,
                )
        A_p_s = np.zeros((self.T, self.sdim))
        A_c = np.zeros((self.T,))
        V_p_s = np.array(self.p_s)
        V_c = np.array(self.c)
        for t in range(self.T):
            prod1 = np.dot(self.P_sa[t], self.mu_a[t])
            A_p_s[t] -= prod1
            V_p_s[t] += prod1
            prod2 = np.dot(np.dot(self.mu_a[t], self.P_aa[t]), self.mu_a[t])
            A_c[t] -= prod2
            V_c[t] += prod2
            prod3 = np.trace(np.dot(self.P_aa[t], self.Sigma_a[t]))
            A_c[t] -= prod3
            V_c[t] += prod3
            prod4 = np.dot(self.p_a[t], self.mu_a[t])
            A_c[t] -= prod4
            V_c[t] += prod4
        self.Acoefs = dict(
                P_ss = np.zeros((self.T, self.sdim, self.sdim)),
                P_sa = self.P_sa,
                P_aa = self.P_aa,
                p_s = A_p_s,
                p_a = self.p_a,
                c = A_c,
                sign = -1,
                )
        self.Vcoefs = dict(
                P_ss = self.P_ss,
                P_sa = np.zeros_like(self.P_sa),
                p_aa = np.zeros_like(self.P_aa),
                p_s = V_p_s,
                p_a = np.zeros_like(self.p_a),
                c = V_c,
                sign = -1.0,
                )

    def compute_P(self):
        assert self.A_tiled and self.Q_tiled and self.R_tiled
        Q = self.Q[0]
        R = self.R[0]
        self.P_ss = np.zeros((self.T, self.sdim, self.sdim))
        self.P_ss.fill(np.nan)
        self.P_ss[-1] = Q # index T-1 is Q
        self.P_aa = np.zeros((self.T, self.adim, self.adim))
        self.P_aa.fill(np.nan)
        self.P_aa[-1] = R # index T-1 is R
        self.P_sa = np.zeros((self.T, self.sdim, self.adim))
        self.P_sa.fill(np.nan)
        self.P_sa[-1].fill(0)
        for i in reversed(range(self.T-1)):
            self.P_ss[i] = self.P_ss[i+1] + self.gamma ** (self.T-1-i) * np.dot(
                    np.dot(self.L[0,self.T-i].T, Q), self.L[0, self.T-i])
            self.P_aa[i] = self.P_aa[i+1] + self.gamma ** (self.T-1-i) * np.dot(
                    np.dot(self.LB[0,self.T-i-1].T, Q), self.LB[0, self.T-i-1])
            self.P_sa[i] = self.P_sa[i+1] + self.gamma ** (self.T-1-i) * np.dot(
                    np.dot(self.L[0,self.T-i].T, Q), self.LB[0, self.T-i-1])
            assert is_sym(self.P_ss[i]), self.P_ss[i]
            assert is_sym(self.P_aa[i]), self.P_aa[i]
        self.P_sa *= 2.0
        assert not np.isnan(self.P_ss).any()
        assert not np.isnan(self.P_sa).any()
        assert not np.isnan(self.P_aa).any()

def getPointMassEnv(sigma_s_sa=0.001, T = 5, dim = 2, r=0.2, q=1.0, m=1.0, sigma_a = 0.01):
    # env parameters
    dt = 0.05 # timestep
    mu_s_0 = [3.0, 4.0, 0.5, -0.5]
    init_mu_mu_a = 0.0
    init_mu_Sigma_a = 0.3
    kwargs = dict(
        T = T,
        gamma = 0.99,
        )

    # derived parameters
    sdim = dim * 2
    adim = dim
    A = np.eye(sdim)
    A[:dim, dim:] += np.eye(dim) * dt
    B = np.zeros((sdim, adim))
    B[dim:] = np.eye(dim) * dt / m
    kwargs.update(dict(
        sdim=sdim,
        adim=adim,
        A=A,
        B=B,
        Q = np.eye(sdim) * q,
        R = np.eye(adim) * r,
        Sigma_s_sa = np.eye(sdim) * sigma_s_sa,
        ))

    # init policies
    mu_a = np.random.randn(T, adim) * np.sqrt(init_mu_Sigma_a) + init_mu_mu_a
    L_a = np.eye(adim) * np.sqrt(sigma_a)
    L_a = np.tile(L_a[None], (T, 1, 1,))
    pi_kwargs = dict(
        mu_a = mu_a,
        L_a = L_a,
        mu_s_0 = np.array(mu_s_0),
        )

    # create env
    env = GlobalLQG(**kwargs)
    env.precompute()
    env.set_pi(**pi_kwargs)
    env.precompute_pi_conditional()
    return env, kwargs, pi_kwargs

def testPointMass():
    trial = 't1'
    lr = 0.001
    mom = 0.1
    ng = False
    ignore_Sigma = True
    T = 100
    sigma_a = 0.0001
    sigma_s_sa = 0.00001
    m = 1.0
    q = 1.0
    r = 0.01
    lambdas = [0.0, 0.99]
    seed = 3
    variance_fn = 'logsumdiag'

    exp_dict = dict(
            trial = trial,
            lr=lr,
            mom=mom,
            ng=ng,
            ignore_Sigma=ignore_Sigma,
            T=T,
            sigma_a=sigma_a,
            sigma_s_sa=sigma_s_sa,
            m=m,
            q=q,
            r=r,
            lambdas=lambdas,
            seed=seed,
            )

    np.random.seed(seed)
    env, env_dict, _ = getPointMassEnv(sigma_s_sa=sigma_s_sa, q=q, r=r, m=m, T=T, sigma_a = sigma_a)

    #env.test_grad_mu(t=5)
    #env.test_rollout(t=5)
    #env.test_value_fns(t=5)
    #sys.exit(0)

    for i in range(50000):
        if i % 10 == 0:
            pi_dict = env.get_pi()
            with open('%s_pi_%04d.pkl' % (trial, i), 'wb') as f:
                pickle.dump(dict(pi=pi_dict, exp=exp_dict, env=env_dict), f)
            bs, _, _, _, _ = env.rollout(t=0, n=3)
            plot_trajs(bs, filename='%s_traj_%04d' % (trial, i), title='%04d' % i)
            variances = env.get_variances(20000, lambdas=lambdas, variance_fn=variance_fn)
            plot_variances(variances, filename='%s_variances_%04d' % (trial,i),
                    title='\#updates=%04d' % i, xlim=None, ylim=None)
        r = env.optimize(ignore_Sigma=ignore_Sigma, opt_params=dict(lr=lr, ng=ng, mom=mom))
        print('[opt] iter %03d: r=%f' % (i, r))
    sys.exit(0)

if __name__ == '__main__':
    testPointMass()
