#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam
from lib.resolution import smear_mdpi
from lib.params import gs, gt

def getdpispec(emin=-2, emax=3):
    """ """
    N = 512
    bins=256
    bins2=512
    E = np.linspace(emin, emax, N)*10**-3
    pdf = [
        DnDnPip(gs, gt, E[-1]),
        DnDnPip(gs, gt, E[-1]),
    ]

    grids = [
        pdf[0].mgridABAC(bins, bins2)[0],
        pdf[0].mgridABAC(bins, bins2)[0],
    ]

    I = [np.zeros(bins2) for _ in grids]

    fcn = [
        lambda: pdf[0].mdpihspec(grid=grids[0]),
        lambda: pdf[0].mdpilspec(grid=grids[0]),
    ]

    labels = [
        r'$D^0\pi^+$ high',
        r'$D^0\pi^+$ low',
    ]

    for idx, energy in enumerate(E):
        print(f'E = {energy*10**3:.3f} MeV ({idx}/{E.shape[0]})')
        for p in pdf:
            p.setE(energy)
        for i, f in zip(I, fcn):
            i += f()

    mdpi = [np.sqrt(g[1][:,0]) for g in grids]
    I = [i*m for i,m in zip(I, mdpi)]  # Jacobian d(m^2) = 2m*dm
    norm = [np.sum(i) * (m[-1] - m[0]) / bins2 for i,m in zip(I, mdpi)]
    I = [i / n for i,n in zip(I, norm)]
    plt.figure(figsize=(8,6))
    for x, y, l in zip(mdpi, I, labels):
        plt.plot(x,y, label=l)
        plt.plot(*smear_mdpi(x,y, 1024), label=l+' smeared')
    plt.ylim(0, 1.01*max([np.max(i) for i in I]))
    plt.xlim(2.00, 2.015)
    plt.xlabel(r'$m(D^0\pi^+)$ (GeV)', fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    try:
        elo, ehi = [float(x) for x in sys.argv[1:]]
        getdpispec(elo, ehi)
    except ValueError:
        print('Usage: ./smeardpispec [E low] [E high]')
