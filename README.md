# The Mirage of Action-Dependent Baselines in Reinforcement Learning

Code accompanying *The Mirage of Action-Dependent Baselines in Reinforcement Learning*. George Tucker, Surya Bhupatiraju, Shixiang Gu, Richard E. Turner, Zoubin Ghahramani, Sergey Levine. ICML 2018. (https://arxiv.org/abs/1802.10031)

## Linear-Quadratic-Gaussian (LQG) systems
See Appendix Section 9 for a detailed description of the LQG system. The code in
this folder was used to generate the results in the LQG section (3.1) and
Figures 1 and 5.

## Q-Prop (https://arxiv.org/abs/1611.02247)
We modified the Q-Prop implementation published
by the authors at https://github.com/
shaneshixiang/rllabplusplus (commit:
4d55f96). For our experiments, we used the conservative variant of QProp,
as is used throughout the experimental section in the
original paper. We used the default choices of policy and
value functions, learning rates, and other hyperparameters. This code was
used to generate Figure 3 and we describe the modifications in detail in Appendix 8.1.


## Backpropagation through the Void (https://arxiv.org/abs/1711.00123)
We used the implementation published by the authors
(https://github.com/wgrathwohl/
BackpropThroughTheVoidRL, commit: 0e6623d)
with the following modification: we measure the variance
of the policy gradient estimator. In the original code, the
authors accidentally measure the variance of a gradient
estimator that neither method uses. We note that Grathwohl
et al. (2018) recently corrected a bug in the code that caused
the LAX method to use a different advantage estimator than
the base method. We use this bug fix. The code was used to generate Figure 13.

## Action-depedent Control Variates for Policy Optimization via Stein's Identity (https://arxiv.org/abs/1710.11198)
We used the Stein control variate implementation published
by the authors at https://github.com/DartML/
PPO-Stein-Control-Variate (commit: 6eec471). We describe the experiments in
Appendix Section 8.2 and use the code to generate Figures 8 and 12.

## Variance calculations
For variance measurement experiments, we modify the open-source
TRPO implementation: https://github.com/
ikostrikov/pytorch-trpo (commit: 27400b8). We used this code to generate Figures
2, 4, 9, 10, and 11.

## Horizon-aware
For the horizon-aware experiments, we modify the open-source
TRPO implementation: https://github.com/
ikostrikov/pytorch-trpo (commit: 27400b8). We used this code to generate Figures
6 and 7.

