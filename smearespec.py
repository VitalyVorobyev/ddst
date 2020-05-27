#! /usr/bin/env python
""" """

import sys
sys.path.append('./lib')

import numpy as np
import matplotlib.pyplot as plt

from params import gs, gt, mdn, mdp
from resolution import smear_e

from dndnpip import DnDnPip
from dndppin import DnDpPin
from dndpgam import DnDpGam

def getespec(emin=-2, emax=4):
    """ """
    N = 1024
    bins=512
    E = np.linspace(emin, emax, N)*10**-3

    pdf = [
        # DnDpPin(gs, gt, E[-1]),
        DnDnPip(gs, gt, E[-1]),
        # DnDpGam(gs, gt, E[-1])
    ]

    labels = [
        # r'$D^0D^+\pi^0$',
        r'$D^0D^0\pi^+$',
        # r'$D^0D^+\gamma$'
    ]

    thr = [
        # mdn + mdp,
        mdn + mdn,
        # mdn + mdp
    ]

    grid = [p.mgridABAC(bins//2, bins) for p in pdf]
    tdds = [np.sqrt(g[0][0][0]) - th for g, th in zip(grid, thr)]
    I = [np.empty(E.shape) for _ in pdf]
    I_smeared = [np.zeros(E.shape) for _ in pdf]
    sigmas = [np.zeros(E.shape) for _ in pdf]

    for idx, energy in enumerate(E):
        for i, ism, p, g, tdd, sig in zip(I, I_smeared, pdf, grid, tdds, sigmas):
            p.setE(energy)
            (mdd, mdh), ds = g
            dens = p(mdd, mdh)[0]
            ptdd = ds * np.sum(dens, axis=0)
            i[idx] = np.sum(ptdd)
            ism_j, sigma = smear_e(energy, E, tdd, ptdd)
            sig[idx] = sigma*10**3
            ism += ism_j

    E *= 10**3
    norm = [np.sum(i) * (E[-1] - E[0]) / N for i in I]
    I = [i/n for i,n in zip(I, norm)]
    # I_smeared = [i/n for i,n in zip(I_smeared, norm)]
    plt.figure(figsize=(8,6))
    for i, ism, l in zip(I, I_smeared, labels):
        plt.plot(E, i, label=l)
        plt.plot(E, ism / np.sum(ism) * np.sum(i), label=l+' smeared')
    plt.xlim(E[0], E[-1])
    plt.ylim(0, 1.05*max([np.max(i) for i in I]))
    plt.legend(loc='best', fontsize=20)
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.tight_layout()
    plt.grid()

    plt.figure(figsize=(6,4))
    plt.plot(E, sigmas[0])
    plt.xlim(E[0], E[-1])
    plt.ylim(0, 1.05*np.max(sigmas[0]))
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.ylabel(r'$\sigma(E), MeV$', fontsize=18)
    plt.tight_layout()
    plt.grid()

    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getespec(elo, ehi)
