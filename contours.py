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
        print(f'E {energy*10**3:.3f} MeV ({idx}/{E.shape[0]})')
        pdf.setE(energy)
        I[:,idx] = np.sum(pdf(*gridABAC)[0], axis=0) * sqrtABspace

    tdd = (sqrtABspace - 2*mdn)

    print('Applying energy resolution...')
    for idx, x in enumerate(tdd):
        I[idx,:] = smear_e_fixed_tdd(E, I[idx,:], x)

    tdd *= 10**3
    E *= 10**3
    x,y = np.meshgrid(tdd, E)

    label_size = 16
    plt.figure(figsize=(7,7))
    plt.contourf(x, y, I.T, levels=25)
    plt.xlabel('T(DD) (MeV)', fontsize=label_size)
    plt.ylabel('E (MeV)', fontsize=label_size)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/e_vs_tdd.{ext}')

    plt.figure(figsize=(7,7))
    plt.contourf(x, y, np.log(1. + I.T + 1.e-6), levels=25)
    plt.xlabel('T(DD) (MeV)', fontsize=label_size)
    plt.ylabel('E (MeV)', fontsize=label_size)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/e_vs_tdd_log.{ext}')
    
    plt.show()


if __name__ == '__main__':
    try:
        elo, ehi = [float(x) for x in sys.argv[1:]]
        run(elo, ehi)
    except ValueError:
        print('Usage: ./countours.py [E low] [E high] (MeV)')
