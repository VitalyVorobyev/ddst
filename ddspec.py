#! /usr/bin/env python
""" """
import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam
from lib.params import gs, gt, mdn, mdp

def tdd(mdd, m1, m2):
    """ Kinetic energy of D0D- in their frame """
    return mdd - m1 - m2

def getddspec(emin=-2, emax=3):
    """ """
    N = 512
    bins=1024
    bins2=512
    E = np.linspace(emin, emax, N)*10**-3
    tfcn = [
        lambda x: tdd(x, mdn, mdn),
        lambda x: tdd(x, mdn, mdp),
        lambda x: tdd(x, mdn, mdp)
    ]
    pdf = [
        DnDnPip(gs, gt, E[-1]),
        DnDpPin(gs, gt, E[-1]),
        DnDpGam(gs, gt, E[-1])
    ]
    I = [np.zeros(bins) for _ in pdf]
    grids = [p.mgridABAC(bins, bins2)[0] for p in pdf]
    mddSq = [p.linspaceAB(bins) for p in pdf]
    labels = [
        r'$D^0D^0(\pi^+)$',
        r'$D^0D^+(\pi^0)$',
        r'$D^0D^+(\gamma)$',
    ]
    print(mddSq)
    for energy in E:
        for i, p, g in zip(I, pdf, grids):
            p.setE(energy)
            i += p.mddspec(grid=g)

    I = [i*np.sqrt(m) for i,m in zip(I, mddSq)]  # Jacobian d(m^2) = 2m*dm
    mdd = [t(np.sqrt(m))*10**3 for m, t in zip(mddSq, tfcn)]
    # norm = [np.sum(i) * (m[-1] - m[0]) / bins for i,m in zip(I, mdd)]
    norm = [p.linspaceAB(bins)[1]-p.linspaceAB(bins)[0] for p in pdf]
    I = [i / n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    for m, i, l in zip(mdd, I, labels):
        plt.plot(m, i, label=l)
    plt.xlim(0, min(15, 1.05*max([i[-1] for i in mdd])))
    plt.ylim(0, 1.01*max([np.max(i) for i in I]))
    plt.xlabel(r'$E(DD)$ (MeV)', fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getddspec(elo, ehi)
