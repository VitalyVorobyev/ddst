#! /usr/bin/env python
""" """

import sys
sys.path.append('./lib')

import numpy as np
import matplotlib.pyplot as plt

from params import gs, gt

from dndnpip import DnDnPip
from dndppin import DnDpPin
from dndpgam import DnDpGam

def getespec(emin=-2, emax=4):
    """ """
    N = 1024
    bins=512
    E = np.linspace(emin, emax, N)*10**-3

    pdf = [
        DnDpPin(gs, gt, E[-1]),
        DnDnPip(gs, gt, E[-1]),
        DnDpGam(gs, gt, E[-1])
    ]

    labels = [
        r'$D^0D^+\pi^0$',
        r'$D^0D^0\pi^+$',
        r'$D^0D^+\gamma$'
    ]

    grid = [p.mgridABAC(bins, bins) for p in pdf]
    I = [np.empty(E.shape) for _ in pdf]

    for idx, energy in enumerate(E):
        for i, p, g in zip(I, pdf, grid):
            p.setE(energy)
            (mdd, mdh), ds = g
            i[idx] = ds * np.sum(p(mdd, mdh))

    E *= 10**3
    norm = [np.sum(i) * (E[-1] - E[0]) / N for i in I]
    I = [i/n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    for i, l in zip(I, labels):
        plt.plot(E, i, label=l)
    plt.xlim(E[0], E[-1])
    plt.ylim(0, 1.05*max([np.max(i) for i in I]))
    plt.legend(loc='best', fontsize=20)
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getespec(elo, ehi)
