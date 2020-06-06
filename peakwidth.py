#! /usr/bin/env python

import sys
sys.path.append('./lib')

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from dndnpip import DnDnPip
from resolution import smear_tdd, smear_e, smear_mdpi
from params import *

def run(elo=-0.5, ehi=0.0):
    """ """
    nEbins = 512
    nABbins = 512
    nACbins = 512

    E = np.linspace(elo, ehi, nEbins)*10**-3
    I = np.zeros(nEbins)
    pdf = DnDnPip(gs, gt, E[-1])
    gridABAC, ds = pdf.mgridABAC(nABbins, nACbins)

    for idx, energy in enumerate(E):
        print(f'E {energy*10**3:.3f} MeV')
        pdf.setE(energy)
        I[idx] = ds*np.sum(pdf(*gridABAC)[0])

    E *= 10**3

    r = UnivariateSpline(E, I-np.max(I)/2, s=0).roots()
    if len(r) > 1:
        r1, r2 = r[:2]
    else:
        r1 = r[0]
        r2 = 0

    plt.figure(figsize=(8, 6))
    plt.title(f'gs {gs.real*10**3:.1f} + i{gs.imag*10**3:.1f} MeV, gs {gt.real*10**3:.1f} + i{gt.imag*10**3:.1f} MeV, fwhw = {0.5*(r2-r1)*10**3:.3f} keV')
    plt.xlabel(r'$E$ (MeV)', fontsize=16)
    plt.ylim=(0, 1.01)
    plt.xlim=(E[0], E[-1])
    plt.plot(E, I / np.max(I))
    plt.axvspan(r1, r2, facecolor='orange', alpha=0.2)
    plt.grid()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    run(elo, ehi)
