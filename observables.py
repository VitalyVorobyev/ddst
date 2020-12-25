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


def init_observs(elo=-2, ehi=8, nEbins=256, nABbins=256, nACbins=256, nBCbins=256):
    """
        Makes a dictionary for observables
        {key : [np.ndarray, callable(E)]}
    """
    E = np.linspace(elo, ehi, nEbins)
    ddpipFull = DnDnPip(gs, gt, E[-1], [True, include_swave])

    ab_space = ddpipFull.linspaceAB(nABbins)
    ac_space = ddpipFull.linspaceAC(nACbins)
    tdd_space = np.sqrt(ab_space) - (mdn + mdn)
    delta_ab = ab_space_ddpip[1] - as_space_ddpip[0]
    delta_ac = ac_space_ddpip[1] - ac_space_ddpip[0]

    gridABAC = ddpipFull.mgridABAC(nABbins, nACbins)[0]
    gridAB, gridAC = gridABAC[0], gridABAC[1]

    gridACBC = ddpipFull.mgridACBC(nACbins, nBCbins)[0]
    gridACp, gridBCp = gridACBC[0], gridACBC[1]
    ddpi_mask = (gridBCp > gridACp)


    def commonCalcDdpipFull(energy):
        ddpipFull.setE(energy)
        mab = delta_ac * ddpipFull(gridAB, gridAC)[0].sum(axis=0)

        gridABcur = ddpipFull.mZsq(gridACp, gridBCp)
        densACBC = ddpipFull(gridABcur, gridACp)[0]
        densACBC[ddpi_mask] = 0
        mac = np.sum(densACBC, axis=0)

        return (mab, mac)

    data_ddpi = np.zeros(nEbins)
    data_ddpis = np.zeros(nEbins)
    data_d0d0 = np.zeros(nABbins)
    data_d0pib = np.zeros(nACbins)
    data_d0pia = np.zeros(nACbins)

    def update_ddpi(idx, mab, mac):
        data_ddpi[idx] = mab.sum() * delta_ab

    def update_d0d0(idx, mab, mac):
        data_d0d0 += mab

    def update_d0pi(idx, mab, mac):
        if E[idx] > 0:
            data_d0pia += mac.sum() * delta_ac
        else:
            data_d0pib += mac.sum() * delta_ac

    def update_ddpis(idx, mab, mac):
        data_ddpis += smear_e(energy, E, tdd_space, mab)[0]]

    obs = [
        ['DDpi', data_ddpi, update_ddpi],
        ['D0D0', data_d0d0, update_d0d0],
        ['D0pi', [data_d0pib, data_d0pia], update_d0pi],
        ['DDpiS', data_ddpis, update_ddpis],
    ]

    return (E, commonCalc, obs, ab_space, tdd_space)


def run(elo=-2, ehi=8):
    """ """
    E, commonCalc, observs, ab_space, tdd_space = init_observs(
        elo=elo, ehi=ehi, nEbins=256, nABbins=256, nACbins=256, nBCbins=256)

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        mab, mac = commonCalc(energy)
        map(lambda x: x[-1](idx, mab, mac), observs)

    observs = {label: data for label, data, _ in  observs.items()}
    observs.update({
        'tDDspace' : tdd_space, 'E' : E,
        'tD0D0' : observs['D0D0'] * 2 * np.sqrt(ab_space),

    })

    return observs


def dndnpi_plot(ax, observs, elo=-2, ehi=15, key='DDpi'):
    ax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(elo, ehi))
    ax.plot(observs('E'), observs[key] / observs[key].max(), label=r'$D^0D^0\pi^+$')


def dndnpis_plot(ax, observs, elo=-2, ehi=15)
    dndnpi_plot(ax, observs, elo=elo, ehi=ehi, key='DDpiS')


def tdd_plot(ax, observs, tddhi):
    tddspace = [x*10**3 for x in tddspace]
    pdndn   = I['D0D0']   * 2 * np.sqrt(abspace[0])
    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, min(ehi*2, 15)))
    cax.plot(tddspace[0], pdndn / ymax, label=r'$D^0D^0$')


def make_observs_plot(observs):
    _, ax = plt.subplots(2, 3, figsize=(18,12))
    for a in ax.ravel():
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    dndnpi_plot(ax[0,0], observs)  # Energy w/o smearing
    dndnpis_plot(ax[1,0], observs)  # Energy w/ smearing
    tdd_plot(ax[0,1], observs)  # T(DD) w/o smearing

    # m(DD) w/ smearing
    cax = ax[1,1]
    for x in tddspace:
        x[0] += 1.e-6
    dndns = smear_tdd(tddspace[0]/10**3, pdndn, dots=1000)
    x, y = merge(*pdndppis, *pdndpgas)
    ymax = max(np.max(dndns[1]), np.max(y))

    cax.set(xlabel=r'$T(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, min(ehi*2, 15)))
    cax.plot(dndns[0], dndns[1] / ymax, label=r'$D^0D^0$')

    # m(Dpi) w/o smearing
    cax = ax[0,2]
    pdnpiAbove = I['D0piAbove'] * 2 * np.sqrt(acspace[0])
    pdnpiBelow = I['D0piBelow'] * 2 * np.sqrt(acspace[0])
    x = np.sqrt(acspace[0])
    ytotal = pdnpiAbove + pdnpiBelow
    ymax = max(ytotal)
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], min(x[-1], 2.02)))
    cax.plot(x, ytotal     / ymax,       label=r'$D^0\pi^+$ high')
    cax.plot(x, pdnpiAbove / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    cax.plot(x, pdnpiBelow / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')

    # m(Dpi) w/ smearing
    cax = ax[1,2]
    cax.set(xlabel=r'$m(D^0\pi^+)$ (GeV)', ylim=(0, 1.01), xlim=(x[0], min(x[-1], 2.02)))
    x0, y0 = smear_mdpi(x, pdnpiAbove, 1024)
    xp, yp = smear_mdpi(x, pdnpiBelow, 1024)
    ytotal = y0 + yp
    ymax = max(ytotal)
    cax.plot(x0, ytotal / ymax,       label=r'$D^0\pi^+$ high')
    cax.plot(x0,    y0  / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    cax.plot(xp,    yp  / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')

    for a in ax.ravel():
        a.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig('plots/obs.png')
    plt.savefig('plots/obs.pdf')
    plt.show()

if __name__ == '__main__':
    try:
        elo, ehi = list(map(float, sys.argv[1:]))
        run(elo, ehi)
    except ValueError:
        print('Usage: ./observables [E low] [E high]')
