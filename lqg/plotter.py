#!python
import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
import matplotlib.patches as mpatches
color_list = sns.color_palette("muted", 10)
sns.palplot(color_list)


def plot_trajs(s, filename, title, ylim=(-2,6), xlim=(-2,6)):
    print(filename, title)
    if os.path.exists(filename + '.npz'):
        s = np.load(filename + '.npz')['s']
    else:
        np.savez(filename, s=s)
    assert s.shape[2] == 4
    plt.figure( figsize=(6,6))
    for i in range(s.shape[1]):
        plt.plot(s[:,i,0], s[:,i,1], zorder=1)
        plt.scatter(s[:,i,0], s[:,i,1], zorder=2)
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.5)
    plt.savefig(filename + '.png')
    plt.close()

def plot_multiple_trajs(filenames, outfile, titles, ylim=(-2,6), xlim=(-2, 6)):
    n = len(filenames)
    dats = []
    for i in range(n):
        dats.append(np.load(filenames[i]+'.npz')['s'])
    m = dats[0].shape[1]
    fig, axes = plt.subplots(1, n, figsize=(20, 3))
    for i in range(n):
        ax = axes[i]
        for j in range(m):
            ax.plot(dats[i][:,j,0], dats[i][:,j,1],color=color_list[j], alpha=0.5, zorder=1, linewidth=0.5)
            ax.scatter(dats[i][:,j,0],dats[i][:,j,1], color=color_list[j], alpha=0.5, zorder=2, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        ax.set_title(titles[i])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.5)
    fig.savefig('%s.pdf' % outfile , bbox_inches='tight', dpi=200, format='pdf')

def plot_multiple_variances(filenames, outfile, titles, ylim=(-10,28), xlim=(0, 100)):
    n = len(filenames)
    dats = []
    for i in range(n):
        d = np.load(filenames[i]+'.npz')['dat']
        dict_ = {k: d.item().get(k) for k in list(d.item().keys())}
        dats.append(dict_)
    keys = sorted(dats[i].keys())
    fig, axes = plt.subplots(1, n, figsize=(20, 3))
    for i in range(n):
        print(i, n, keys, len(color_list))
        ax = axes[i]
        for j, key in enumerate(keys):
            ax.plot(dats[i][key], label=key, color=color_list[j])
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        ax.set_xlabel('time horizon (t)')
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel(r'$\log |\Sigma|$')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.5)
    hs = []
    for j, key in enumerate(keys):
        hs.append(mpatches.Patch(color=color_list[j], label=key))
    leg = fig.legend(handles=hs, loc='lower center', ncol=6, prop={'size': 14})
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 += -0.30
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    fig.savefig('%s.pdf' % outfile , bbox_inches='tight', dpi=200, format='pdf')
    plt.close()

def plot_variances(dat, filename, title, ylim=(-10,28), xlim=(0,100)):
    print(filename, title)
    if dat == None and os.path.exists(filename + '.npz'):
        dat_ = np.load(filename + '.npz')['dat']
        dat = dict()
        for k in sorted(list(dat_.item().keys())):
            dat[k] = dat_.item().get(k)
    else:
        np.savez(filename, dat=dat)
    sorted_keys = sorted(dat.keys())
    plt.figure(figsize=(6,6))
    for i, key in enumerate(sorted_keys):
        plt.plot(dat[key], label=key, linewidth=2.0, color=color_list[i])
    plt.legend(loc='lower left')
    plt.title(title)
    plt.ylabel(r'$\log |\Sigma|$')
    plt.xlabel('time horizon (t)')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(alpha=0.5)
    plt.savefig('%s.png' % filename, bbox_inches='tight')
    plt.savefig('%s.pdf' % filename, bbox_inches='tight', dpi=200, format='pdf')
    plt.close()

if __name__ == '__main__':

    #fileids = [0, 150, 300, 600]
    #fileids = list(range(0, 700, 150)) # plot first up to first 1000 updates
    fileids = list(range(0, 1000, 10)) # plot first up to first 1000 updates
    trial = 't1'
    titles = ['\#updates=%04d' % i for i in fileids]
    filenames = ['%s_variances_%04d' % (trial, i) for i in fileids] # variance plots
    #filenames = ['%s_traj_%04d' % (trial, i) for i in fileids] # traj plots
    outfile = 'out_%s' % trial # variance plots
    #outfile = 'out_%s_traj' % trial # traj plots
    #plot_multiple_variances(filenames, outfile, titles) # variance plots
    #plot_multiple_trajs(filenames, outfile, titles) # traj plots
    for f, t in zip(filenames, titles):
        plot_variances(dat=None, filename=f, title=t) # variance plots
        #plot_trajs(s=None, filename=f, title=t) # traj plots

