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

def getespec(emin=3, emax=10):
    """ """
    N = 512
    bins=256
    E = np.linspace(emin, emax, N)*10**-3

    pdf = [
        DnDnPip(gs, gt, E[-1]),
        DnDpPin(gs, gt, E[-1], channels=[True, False]),
        DnDpPin(gs, gt, E[-1], channels=[False, True]),
        DnDpGam(gs, gt, E[-1])
    ]

    # labels = [
    #     r'$D^0[D^{*+}\to D^0\pi^+]$',
    #     r'$D^0[D^{*+}\to D^+\pi^0]$',
    #     r'$D^+[D^{*0}\to D^0\pi^0]$',
    #     r'$D^+[D^{*0}\to D^0\gamma]$'
    # ]

    grid = [p.mgridABAC(bins, bins)[0] for p in pdf]
    I = [np.zeros(N) for _ in pdf]

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        for i, p, g in zip(I, pdf, grid):
            p.setE(energy)
            # p.t1, p.t2, p.t = 1, 1, 1
            i[idx] = p.integral(grid=g)

    E *= 10**3
    plt.figure(figsize=(8,6))
    # for i, l in zip(I, labels):
        # print('{}: {:.3f}'.format(l, np.sum(i)))
        # plt.plot(E, i, label=l)
    plt.plot(E, I[0] / I[1], label=r'$D^{*+}\to D^0\pi^+$ / $D^{*+}\to D^+\pi^0$')
    plt.plot(E, I[3] / I[2], label=r'$D^{*0}\to D^0\gamma$ / $D^{*0}\to D^0\pi^0$')
    plt.plot(E, I[2] / I[0], label=r'$D^{*0}\to D^0\pi^0$ / $D^{*+}\to D^0\pi^+$')
    plt.xlim(E[0], E[-1])
    # plt.ylim(0, 1.05*max([np.max(i) for i in I]))
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getespec(elo, ehi)
