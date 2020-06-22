#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.params import gs, gt
from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam


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

    grid = [p.mgridABAC(bins, bins)[0] for p in pdf]
    I = [np.empty(E.shape) for _ in pdf]

    for idx, energy in enumerate(E):
        print(f'E = {energy*10**3:.3f} ({idx}/{E.shape[0]})')
        for i, p, g in zip(I, pdf, grid):
            p.setE(energy)
            i[idx] = p.integral(grid=g)

    E *= 10**3
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
    try:
        elo, ehi = [float(x) for x in sys.argv[1:]]
        getespec(elo, ehi)
    except ValueError:
        print('Usage: ./espec.py [E low] [E high] (MeV)')
