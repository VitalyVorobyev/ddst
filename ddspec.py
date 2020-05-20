#! /usr/bin/env python
""" """

import sys
sys.path.append('./lib')

import numpy as np
import matplotlib.pyplot as plt

from dndnpip import DnDnPip
from dndppin import DnDpPin
from params import *

def tdd(mdd, m1, m2):
    """ Kinetic energy of D0D- in their frame """
    return mdd - m1 - m2

def getddspec(gs, gt, emin=-2, emax=3):
    """ """
    N = 1000
    bins=1000
    bins2=1000
    E = np.linspace(emin, emax, N)*10**-3
    I = [np.zeros(bins), np.zeros(bins)]
    tfcn = [lambda x: tdd(x, mdn, mdn), lambda x: tdd(x, mdn, mdp)]
    pdf = [DnDnPip(gs, gt, E[-1]), DnDpPin(gs, gt, E[-1])]
    grids = [p.mgridABAC(bins, bins2)[0] for p in pdf]
    mddSq = [g[0][0] for g in grids]
    print(mddSq)
    for energy in E:
        for idx, p in enumerate(pdf):
            p.setE(energy)
            I[idx] += np.sum(p(*grids[idx])[0], axis=0)

    I = [i*np.sqrt(m) for i,m in zip(I, mddSq)]  # Jacobian d(m^2) = 2m*dm
    mdd = [t(np.sqrt(m))*10**3 for m, t in zip(mddSq, tfcn)]
    norm = [np.sum(i) * (m[-1] - m[0]) / bins for i,m in zip(I, mdd)]
    I = [i / n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    plt.plot(mdd[0], I[0], label=r'$D^0D^0$')
    plt.plot(mdd[1], I[1], label=r'$D^0D^+$')
    plt.xlim(0, 1.05*max([i[-1] for i in mdd]))
    plt.ylim(0, 1.01*max([np.max(i) for i in I]))
    plt.xlabel(r'$E(DD)$ (MeV)', fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()

def getdpispec(gs, gt, emin=-2, emax=3):
    """ """
    N = 1024
    bins=1024
    bins2=1024
    E = np.linspace(emin, emax, N)*10**-3
    I = [np.zeros(bins2) for _ in range(3)]
    pdf = [DnDnPip(gs, gt, E[-1]), DnDpPin(gs, gt, E[-1])]

    grids = [
        pdf[0].mgridABAC(bins, bins2)[0],
        pdf[1].mgridABAC(bins, bins2)[0],
        pdf[1].mgridACBC(bins, bins2)[0]
    ]

    fcn = [
        lambda: pdf[0](*grids[0])[0],
        lambda: pdf[1](*grids[1])[0],
        lambda: pdf[1](pdf[1].mZsq(*grids[-1]), grids[2][0])[0]
    ]

    labels = [
        r'$D^0\pi^+$',
        r'$D^0\pi^0$',
        r'$D^+\pi^0$'
    ]

    for energy in E:
        for p in pdf:
            p.setE(energy)
        for i, f in zip(I, fcn):
            i += np.sum(f(), axis=1)

    mdpi = [np.sqrt(g[1][:,0]) for g in grids]
    I = [i*m for i,m in zip(I, mdpi)]  # Jacobian d(m^2) = 2m*dm
    norm = [np.sum(i) * (m[-1] - m[0]) / bins2 for i,m in zip(I, mdpi)]
    I = [i / n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    for x, y, l in zip(mdpi, I, labels):
        plt.plot(x,y, label=l)
    plt.ylim(0, 1.01*max([np.max(i) for i in I]))
    plt.xlabel(r'$m(D\pi)$ (GeV)', fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    gs = (30 + 1j) * 10**-3
    gt = (-30 + 1j) * 10**-3
    # getddspec(gs, gt, elo, ehi)
    getdpispec(gs, gt, elo, ehi)
