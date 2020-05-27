#! /usr/bin/env python
""" m(D0D0pi+)
    m(D0D0): D0[D*+ -> D0 pi+]
    m(D0pi+)
    m(D0D+): D0[D*+ -> D+ pi0], D+[D*0 -> D0 pi0], D+[D*0 -> D0 gamma]
"""

import sys
sys.path.append('./lib')

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from dndnpip import DnDnPip
from dndppin import DnDpPin
from dndpgam import DnDpGam
from resolution import smear_tdd, smear_e, smear_mdpi
from params import *

def merge(x1, y1, x2, y2, bins=5000):
    newx = np.linspace(min(x1[0], x2[0]), max(x1[-1], x2[-1]), bins)

    applo1 = newx[newx<x1[0]]
    x1 = np.append(applo1, x1)
    y1 = np.append(np.zeros(applo1.shape), y1)

    applo2 = newx[newx<x2[0]]
    x2 = np.append(applo2, x2)
    y2 = np.append(np.zeros(applo2.shape), y2)

    apphi1 = newx[newx>x1[-1]]
    x1 = np.append(x1, apphi1)
    y1 = np.append(y1, np.zeros(apphi1.shape))

    apphi2 = newx[newx>x2[-1]]
    x2 = np.append(x2, apphi2)
    y2 = np.append(y2, np.zeros(apphi2.shape))

    y1new = interp1d(x1, y1, kind='cubic')(newx)
    y2new = interp1d(x2, y2, kind='cubic')(newx)
    return (newx, y1new+y2new)

def run(elo=-2, ehi=8, peak=[-0.0005, 0.0010]):
    """ """
    nEbins  = 256
    nABbins = 512
    nACbins = 256

    E = np.linspace(elo, ehi, nEbins)*10**-3
    pdf = [
        DnDnPip(gs, gt, E[-1]),
        DnDpPin(gs, gt, E[-1]),
        DnDpGam(gs, gt, E[-1])
    ]

    I = {
        'DDpi'    : np.zeros(nEbins),
        'DDpi0'   : np.zeros(nEbins),
        'DDpiS'   : np.zeros(nEbins),
        'DDpi0S'  : np.zeros(nEbins),
        'D0D+pi'  : np.zeros(nABbins),
        'D0D+ga'  : np.zeros(nABbins),
        'D0D0'    : np.zeros(nABbins),
        'D0pi'    : np.zeros(nACbins),
        'D0piPeak': np.zeros(nACbins),
    }

    grids   = [p.mgridABAC(nABbins, nACbins)[0] for p in pdf]
    abspace = [p.linspaceAB(nABbins) for p in pdf]
    acspace = [p.linspaceAC(nACbins) for p in pdf]
    thrs = [mdn+mdn, mdn+mdp, mdn+mdp]
    tddspace = [np.sqrt(x) - thr for x, thr in zip(abspace, thrs)]
    dab = [x[1] - x[0] for x in abspace]
    dac = [x[1] - x[0] for x in acspace]

    ddpi_mask = (grids[0][1] > pdf[0].mZsq(*grids[0]))
    print(np.sum(ddpi_mask))
    print(ddpi_mask.shape[0]*ddpi_mask.shape[1])

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        for p in pdf:
            p.setE(energy)
        dens = [p(*g)[0] for p, g in zip(pdf, grids)]
        mab = [dx * np.sum(x, axis=0) for x, dx in zip(dens, dac)]
        I['DDpi'][idx] = np.sum(mab[0]) * dab[0]
        I['DDpi0'][idx] = np.sum(mab[1]) * dab[1]
        I['D0D0'] += mab[0]
        I['D0D+pi'] += mab[1]
        I['D0D+ga'] += mab[2]
        dens[0][ddpi_mask] = 0
        I['D0pi'] += dac[0] * np.sum(dens[0], axis=1)
        if (energy >= peak[0]) and (energy <=peak[1]):
            I['D0piPeak'] += dac[0] * np.sum(dens[0], axis=1)
        print(I['D0pi'][200:210])
        print(I['D0piPeak'][200:210])
        I['DDpiS']  += smear_e(energy, E, tddspace[0], mab[0])[0]
        I['DDpi0S'] += smear_e(energy, E, tddspace[1], mab[1])[0]

    fig, ax = plt.subplots(2, 3, figsize=(18,12))
    E *= 10**3
    # Energy w/o smearing
    cax = ax[0,0]
    cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], E[-1]))
    cax.grid()
    cax.plot(E, I['DDpi']  / np.max(I['DDpi']), label=r'$D^0D^0\pi^+$')
    cax.plot(E, I['DDpi0'] / np.max(I['DDpi']), label=r'$D^0D^+\pi^0$')
    cax.legend(loc='best', fontsize=16)

    # Energy w/ smearing
    cax = ax[1,0]
    cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], E[-1]))
    cax.grid()
    cax.plot(E, I['DDpiS']  / np.max(I['DDpiS']), label=r'$D^0D^0\pi^+$')
    cax.plot(E, I['DDpi0S'] / np.max(I['DDpiS']), label=r'$D^0D^+\pi^0$')
    cax.legend(loc='best', fontsize=16)

    # m(DD) w/o smearing
    cax = ax[0,1]
    tddspace = [x*10**3 for x in tddspace]
    pdndn   = I['D0D0']   * 2 * np.sqrt(abspace[0])
    pdndppi = I['D0D+pi'] * 2 * np.sqrt(abspace[1])
    pdndpga = I['D0D+ga'] * 2 * np.sqrt(abspace[2])
    x, y = merge(tddspace[1], pdndppi, tddspace[2], pdndpga)
    ymax = max(np.max(pdndn), np.max(y))

    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, 8))
    cax.plot(tddspace[0], pdndn   / ymax,        label=r'$D^0D^0$')
    cax.plot(tddspace[1], pdndppi / ymax, ':',   label=r'$D^0D^+(\pi^0)$')
    cax.plot(tddspace[2], pdndpga / ymax, '--' , label=r'$D^0D^+(\gamma)$')
    cax.plot(x, y / ymax, label=r'$D^0D^+ total$')

    cax.grid()
    cax.legend(loc='best', fontsize=16)
    
    # m(DD) w/ smearing
    cax = ax[1,1]
    for x in tddspace:
        x[0] += 1.e-6
    dndns    = smear_tdd(tddspace[0]/10**3, pdndn,   dots=1000)
    pdndppis = smear_tdd(tddspace[1]/10**3, pdndppi, dots=1000)
    pdndpgas = smear_tdd(tddspace[2]/10**3, pdndpga, dots=1000)
    x, y = merge(*pdndppis, *pdndpgas)
    ymax = max(np.max(dndns[1]), np.max(y))

    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, 8))
    cax.plot(dndns[0]*10**3, dndns[1]/ymax,              label=r'$D^0D^0$')
    cax.plot(pdndppis[0]*10**3, pdndppis[1]/ymax, ':',   label=r'$D^0D^+(\pi^0)$')
    cax.plot(pdndpgas[0]*10**3, pdndpgas[1]/ymax, '--' , label=r'$D^0D^+(\gamma)$')
    cax.plot(x*10**3, y/ymax, label=r'$D^0D^+ total$')

    cax.grid()
    cax.legend(loc='best', fontsize=16)

    # m(Dpi) w/o smearing
    cax = ax[0,2]
    pdnpi = I['D0pi'] * 2 * np.sqrt(acspace[0])
    pdnpiPeak = I['D0piPeak'] * 2 * np.sqrt(acspace[0])
    x = np.sqrt(acspace[0])
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], x[-1]))
    cax.plot(x, pdnpi / np.max(pdnpi), label=r'$D^0\pi^+$ high')
    cax.plot(x, pdnpiPeak / np.max(pdnpi), label=r'$D^0\pi^+$ high peak')
    cax.grid()
    cax.legend(loc='best', fontsize=16)

    # m(Dpi) w/ smearing
    cax = ax[1,2]
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], x[-1]))
    x0, y0 = smear_mdpi(x, pdnpi, 1024)
    xp, yp = smear_mdpi(x, pdnpiPeak, 1024)
    cax.plot(x0, y0 / np.max(y0), label=r'$D^0\pi^+$ high')
    cax.plot(xp, yp / np.max(y0), label=r'$D^0\pi^+$ high peak')
    cax.grid()
    cax.legend(loc='best', fontsize=16)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    run(elo, ehi)
    