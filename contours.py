#! /usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.params import gs, gt, mdn
from lib.resolution import smear_e_fixed_tdd

def run(elo=-1, ehi=1):
    """ """
    nEbins  = 256
    nABbins = 512
    nACbins = 256

    E = np.linspace(elo, ehi, nEbins)*10**-3
    pdf = DnDnPip(gs, gt, E[-1], [True, True])

    gridABAC = pdf.mgridABAC(nABbins, nACbins)[0]
    abspace = pdf.linspaceAB(nABbins)
    sqrtABspace = np.sqrt(abspace)
    E = np.linspace(elo, ehi, nEbins)*10**-3
    I = np.empty((nABbins, nEbins))

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        pdf.setE(energy)
        I[:,idx] = np.sum(pdf(*gridABAC)[0], axis=0) * sqrtABspace

    tdd = (sqrtABspace - 2*mdn)

    for idx, x in enumerate(tdd):
        I[idx,:] = smear_e_fixed_tdd(E, I[idx,:], x)

    tdd *= 10**3
    E *= 10**3
    x,y = np.meshgrid(tdd, E)

    plt.figure(figsize=(7,7))
    plt.contourf(x, y, I.T, levels=25)
    plt.xlabel('T(DD) (MeV)', fontsize=14)
    plt.ylabel('E (MeV)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/e_vs_tdd.png')

    plt.figure(figsize=(7,7))
    plt.contourf(x, y, np.log(1. + I.T + 1.e-6), levels=25)
    plt.xlabel('T(DD) (MeV)', fontsize=14)
    plt.ylabel('E (MeV)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/e_vs_tdd_log.png')
    
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    run(elo, ehi)
