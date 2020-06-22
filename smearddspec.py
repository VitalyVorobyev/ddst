#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam
from lib.resolution import smear_tdd
from lib.params import gs, gt, mdn, mdp

def getddspec(emin=-2, emax=3):
    """ """
    N = 512
    bins=256
    bins2=512
    E = np.linspace(emin, emax, N)*10**-3
    pdf = [
        # DnDnPip(gs, gt, E[-1]),
        DnDpPin(gs, gt, E[-1]),
        # DnDpGam(gs, gt, E[-1])
    ]
    I = [np.zeros(bins) for _ in pdf]
    grids = [p.mgridABAC(bins, bins2) for p in pdf]
    mddSq = [p.linspaceAB(bins) for p in pdf]
    fcns = [lambda: p.mddspec(grid=g[0]) for p, g in zip(pdf, grids)]
    labels = [
        # r'$D^0D^0$',
        r'$D^0D^+$',
        # r'$D^0D^+ (\gamma)$',
    ]
    for energy in E:
        for i, p, f in zip(I, pdf, fcns):
            p.setE(energy)
            i += f()

    I = [i*np.sqrt(m) for i,m in zip(I, mddSq)]  # Jacobian d(m^2) = 2m*dm

    tfcn = [
        # lambda x: x - mdn - mdn,
        lambda x: x - mdn - mdp,
        # lambda x: x - mdn - mdp
    ]
    tdd = [t(np.sqrt(m))*10**3 for m, t in zip(mddSq, tfcn)]
    norm = [np.sum(i) * (m[-1] - m[0]) / bins for i,m in zip(I, tdd)]
    I = [i / n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    for m, i, l in zip(tdd, I, labels):
        m = m[1:]
        i = i[1:]
        plt.plot(m, i, label=l)
        sx, sy = smear_tdd(m/10**3, i, m.shape[0])
        plt.plot(sx*10**3, sy, label=l+' smeared')
    plt.xlim(0, min(15, 1.05*max([i[-1] for i in tdd])))
    plt.ylim(0, 1.01*max([np.max(i) for i in I]))
    plt.xlabel(r'$E(DD)$ (MeV)', fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getddspec(elo, ehi)
