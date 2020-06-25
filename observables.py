#! /usr/bin/env python
""" m(D0D0pi+)
    m(D0D0): D0[D*+ -> D0 pi+]
    m(D0pi+)
    m(D0D+): D0[D*+ -> D+ pi0], D+[D*0 -> D0 pi0], D+[D*0 -> D0 gamma]
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam
from lib.resolution import smear_tdd, smear_e, smear_mdpi, smear_e_const
from lib.params import gs, gt, mdn, mdp

include_pwave = False
include_swave = False
hide_invisible = True

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

def run(elo=-2, ehi=8):
    """ """
    nEbins  = 1024
    nABbins = 1024
    nACbins = 1024
    nBCbins = 1024

    E = np.linspace(elo, ehi, nEbins)*10**-3
    pdf = [
        DnDnPip(gs, gt, E[-1], [True, include_swave]),
        DnDpPin(gs, gt, E[-1], [True, True, include_pwave]),
        DnDpGam(gs, gt, E[-1]),
        DnDpPin(gs, gt, E[-1], [False, False, True]),  # P-wave amplitude
        DnDnPip(gs, gt, E[-1], [False, True]),  # S-wave amplitude
    ]

    I = {
        'Pwave'     : np.zeros(nEbins),
        'Swave'     : np.zeros(nEbins),
        'DDpi'      : np.zeros(nEbins),
        'DDpi0'     : np.zeros(nEbins),
        'DDgam'     : np.zeros(nEbins),
        'DDpiS'     : np.zeros(nEbins),
        'DDpi0S'    : np.zeros(nEbins),
        'DDgamS'    : np.zeros(nEbins),
        'PwaveS'    : np.zeros(nEbins),
        'SwaveS'    : np.zeros(nEbins),
        'D0D+Pw'    : np.zeros(nABbins),
        'D0D0Sw'    : np.zeros(nABbins),
        'D0D+pi'    : np.zeros(nABbins),
        'D0D+ga'    : np.zeros(nABbins),
        'D0D0'      : np.zeros(nABbins),
        'D0piBelow' : np.zeros(nACbins),
        'D0piAbove' : np.zeros(nACbins),
    }

    gridsABAC = [p.mgridABAC(nABbins, nACbins)[0] for p in pdf]
    gridsACBC = [p.mgridACBC(nACbins, nBCbins)[0] for p in pdf]
    abspace = [p.linspaceAB(nABbins) for p in pdf]
    acspace = [p.linspaceAC(nACbins) for p in pdf]
    bcspace = [p.linspaceBC(nBCbins) for p in pdf]
    thrs = [mdn+mdn, mdn+mdp, mdn+mdp, mdn+mdp, mdn+mdn]
    tddspace = [np.sqrt(x) - thr for x, thr in zip(abspace, thrs)]
    dab = [x[1] - x[0] for x in abspace]
    dac = [x[1] - x[0] for x in acspace]
    dbc = [x[1] - x[0] for x in bcspace]

    ddpi_mask = (gridsACBC[0][1] > gridsACBC[0][0])

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        for p in pdf:
            p.setE(energy)
        dens = [p(*g)[0] for p, g in zip(pdf, gridsABAC)]
        mab = [dx * np.sum(x, axis=0) for x, dx in zip(dens, dac)]
        I['DDpi'][idx]  = np.sum(mab[0]) * dab[0]
        I['DDpi0'][idx] = np.sum(mab[1]) * dab[1]
        I['DDgam'][idx] = np.sum(mab[2]) * dab[2]
        if include_pwave:
            I['Pwave'][idx] = np.sum(mab[3]) * dab[3]
        I['Swave'][idx] = np.sum(mab[4]) * dab[4]

        I['D0D0']   += mab[0]
        I['D0D+pi'] += mab[1]
        I['D0D+ga'] += mab[2]
        I['D0D+Pw'] += mab[3]
        I['D0D0Sw'] += mab[4]

        densACBC = pdf[0](pdf[0].mZsq(*gridsACBC[0]), gridsACBC[0][1])[0]
        densACBC[ddpi_mask] = 0
        if energy > 0:
            I['D0piAbove'] += dbc[0] * np.sum(densACBC, axis=0)
        else:
            I['D0piBelow'] += dbc[0] * np.sum(densACBC, axis=0)
        
        # eddpi_smeared, meanres = smear_e(energy, E, tddspace[0], mab[0])
        # print(f'sigma(E) = {meanres*10**3:.3} MeV')
        # I['DDpiS']  += I['DDpi'][idx]*eddpi_smeared
        # I['DDpiS']  += I['DDpi'][idx]*smear_e_const(energy, E, tddspace[0], mab[0])[0]
        I['DDpiS']  += smear_e(energy, E, tddspace[0], mab[0])[0]
        I['DDpi0S'] += smear_e(energy, E, tddspace[1], mab[1])[0]
        # I['DDpi0S'] += I['DDpi0'][idx]*smear_e_const(energy, E, tddspace[1], mab[1])[0]
        # I['DDgamS'] += smear_e(energy, E, tddspace[2], mab[2])[0]
        # if include_pwave:
            # I['PwaveS'] += smear_e(energy, E, tddspace[3], mab[3])[0]
        # I['SwaveS'] += smear_e(energy, E, tddspace[4], mab[4])[0]

    _, ax = plt.subplots(2, 3, figsize=(18,12))
    E *= 10**3
    # Energy w/o smearing
    cax = ax[0,0]
    print('Pwave integral {:.0f}'.format(np.sum(I['Pwave'])))
    print('Swave integral {:.0f}'.format(np.sum(I['Swave'])))
    print(' DDpi integral {:.0f}'.format(np.sum(I['DDpi'])))
    print(' Pwave fraction {:.2f}'.format(np.sum(I['Pwave']) / np.sum(I['DDpi'])))
    print(' Swave fraction {:.2f}'.format(np.sum(I['Swave']) / np.sum(I['DDpi'])))
    ymax = np.max(I['DDpi'])
    # cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], E[-1]))
    cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], 15))
    cax.grid()
    cax.plot(E, I['DDpi']  / ymax, label=r'$D^0D^0\pi^+$')
    cax.plot(E, I['DDpi0'] / ymax, label=r'$D^0D^+\pi^0$')
    cax.plot(E, I['DDgam'] / ymax, label=r'$D^0D^+\gamma$')
    # if not hide_invisible:
    #     cax.plot(E, I['Swave'] / ymax, label=r'$D^0D^0$ $S$-wave')
    #     if include_pwave:
    #         cax.plot(E, I['Pwave'] / ymax, label=r'$D^0D^+$ $P$-wave')
    cax.legend(loc='best', fontsize=16)

    # Energy w/ smearing
    cax = ax[1,0]
    ymax = np.max(I['DDpiS'])
    # cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], E[-1]))
    cax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(E[0], 15))
    cax.grid()
    cax.plot(E, I['DDpiS']  / ymax, label=r'$D^0D^0\pi^+$')
    cax.plot(E, I['DDpi0S'] / ymax, label=r'$D^0D^+\pi^0$')
    if not hide_invisible:
        # cax.plot(E, I['DDgamS'] / ymax, label=r'$D^0D^+\gamma$')
        cax.plot(E, I['SwaveS'] / ymax, label=r'$D^0D^0$ $S$-wave')
        if include_pwave:
            cax.plot(E, I['PwaveS'] / ymax, label=r'$D^0D^+$ $P$-wave')
    cax.legend(loc='best', fontsize=16)

    # m(DD) w/o smearing
    cax = ax[0,1]
    tddspace = [x*10**3 for x in tddspace]
    pdndn   = I['D0D0']   * 2 * np.sqrt(abspace[0])
    pdndppi = I['D0D+pi'] * 2 * np.sqrt(abspace[1])
    pdndpga = I['D0D+ga'] * 2 * np.sqrt(abspace[2])
    pdndppw = I['D0D+Pw'] * 2 * np.sqrt(abspace[3])
    pdndnsw = I['D0D0Sw'] * 2 * np.sqrt(abspace[4])
    x, y = merge(tddspace[1], pdndppi, tddspace[2], pdndpga)
    ymax = max(np.max(pdndn), np.max(y))

    # cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, ehi*2))
    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, 15))
    cax.plot(tddspace[0], pdndn   / ymax,        label=r'$D^0D^0$')
    cax.plot(tddspace[1], pdndppi / ymax, ':',   label=r'$D^0D^+(\pi^0)$')
    cax.plot(tddspace[2], pdndpga / ymax, '--' , label=r'$D^0D^+(\gamma)$')
    cax.plot(tddspace[3], pdndppw / ymax, '--' , label=r'$D^0D^+$ $P$-wave')
    cax.plot(tddspace[4], pdndnsw / ymax, '--' , label=r'$D^0D^0$ $S$-wave')
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
    pdndppws = smear_tdd(tddspace[3]/10**3, pdndppw, dots=1000)
    x, y = merge(*pdndppis, *pdndpgas)
    # x, y = merge(*pdndppws, x, y)
    ymax = max(np.max(dndns[1]), np.max(y))

    # cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, ehi*2))
    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, 15))
    cax.plot(dndns[0]*10**3, dndns[1]/ymax,              label=r'$D^0D^0$')
    cax.plot(pdndppis[0]*10**3, pdndppis[1]/ymax, ':',   label=r'$D^0D^+(\pi^0)$')
    cax.plot(pdndpgas[0]*10**3, pdndpgas[1]/ymax, '--' , label=r'$D^0D^+(\gamma)$')
    if include_pwave:
        cax.plot(pdndppws[0]*10**3, pdndppws[1]/ymax, '--' , label=r'$D^0D^+$ $P$-wave')
    cax.plot(x*10**3, y/ymax, label=r'$D^0D^+ total$')

    cax.grid()
    cax.legend(loc='best', fontsize=16)

    # m(Dpi) w/o smearing
    cax = ax[0,2]
    pdnpiAbove = I['D0piAbove'] * 2 * np.sqrt(acspace[0])
    pdnpiBelow = I['D0piBelow'] * 2 * np.sqrt(acspace[0])
    x = np.sqrt(acspace[0])
    ytotal = pdnpiAbove + pdnpiBelow
    ymax = max(ytotal)
    # cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], x[-1]))
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], 2.02))
    cax.plot(x, ytotal     / ymax,       label=r'$D^0\pi^+$ high')
    cax.plot(x, pdnpiAbove / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    cax.plot(x, pdnpiBelow / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')
    cax.grid()
    cax.legend(loc='best', fontsize=16)

    # m(Dpi) w/ smearing
    cax = ax[1,2]
    # cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], x[-1]))
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], 2.02))
    x0, y0 = smear_mdpi(x, pdnpiAbove, 1024)
    xp, yp = smear_mdpi(x, pdnpiBelow, 1024)
    ytotal = y0 + yp
    ymax = max(ytotal)
    cax.plot(x0, ytotal / ymax,       label=r'$D^0\pi^+$ high')
    cax.plot(x0,    y0  / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    cax.plot(xp,    yp  / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')
    cax.grid()
    cax.legend(loc='best', fontsize=16)

    plt.tight_layout()
    plt.savefig('plots/obs.png')
    plt.savefig('plots/obs.pdf')
    plt.show()

if __name__ == '__main__':
    try:
        elo, ehi = [float(x) for x in sys.argv[1:]]
        run(elo, ehi)
    except ValueError:
        print('Usage: ./observables [E low] [E high]')
