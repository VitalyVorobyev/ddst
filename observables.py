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
from lib.resolution import smear_tdd, smear_e, smear_mdpi, smear_e_const, merge
from lib.params import gs, gt, mdn, mdp

include_swave = False

def init_observs(elo=-2, ehi=8, nEbins=256, nABbins=256, nACbins=256, nBCbins=256):
    """ """
    E = np.linspace(elo, ehi, nEbins)
    ddpipFull = DnDnPip(gs, gt, E[-1], [True, include_swave])

    ab_space = ddpipFull.linspaceAB(nABbins)
    ac_space = ddpipFull.linspaceAC(nACbins)
    tdd_space = np.sqrt(ab_space) - 2*mdn
    delta_ab = ab_space[1] - ab_space[0]
    delta_ac = ac_space[1] - ac_space[0]

    gridABAC = ddpipFull.mgridABAC(nABbins, nACbins)[0]
    gridAB, gridAC = gridABAC[0], gridABAC[1]

    gridACBC = ddpipFull.mgridACBC(nACbins, nBCbins)[0]
    gridACp, gridBCp = gridACBC[0], gridACBC[1]
    ddpi_mask = (gridBCp > gridACp)

    def processEnergy(energy):
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
        data_ddpis += smear_e(energy, E, tdd_space, mab)[0]

    obs = [
        ['DDpi', data_ddpi, update_ddpi],
        ['D0D0', data_d0d0, update_d0d0],
        ['D0pi', [data_d0pib, data_d0pia], update_d0pi],
        ['DDpiS', data_ddpis, update_ddpis],
    ]

    return (E, processEnergy, obs, ab_space, ac_space, tdd_space)


def run(elo=-2, ehi=8):
    """ """
    E, processEnergy, observs, ab_space, ac_space, tdd_space = init_observs(
        elo=elo, ehi=ehi, nEbins=256, nABbins=256, nACbins=256, nBCbins=256)

    for idx, energy in enumerate(E):
        print('E {:.3f} MeV'.format(energy*10**3))
        mab, mac = processEnergy(energy)
        map(lambda x: x[-1](idx, mab, mac), observs)

    observs = {label: data for label, data, _ in  observs}
    
    pdndn = observs['D0D0'] * 2 * np.sqrt(ab_space)
    pdd_space = np.sqrt(tdd_space * mdn)
    mdpi_space = np.sqrt(ac_space)
    mdpi_below = observs['D0pi'][0] * 2 * mdpi_space
    mdpi_above = observs['D0pi'][1] * 2 * mdpi_space
    mdpis_space, mdpiS_below = smear_mdpi(mdpi_space, mdpi_below, 1024)
    _, mdpiS_above = smear_mdpi(mdpi_space, mdpi_above, 1024)
    tdds_space, pdds = smear_tdd(tdd_space, pdndn, dots=1000)
    pdds_space = np.sqrt(tdds_space * mdn)

    observs.update({
        'tD0D0space' : tdd_space,
        'tD0D0Sspace' : tdds_space,
        'pD0D0space' : pdd_space,
        'pD0D0Sspace' : pdds_space,
        'mD0pi_space': mdpi_space,
        'mD0piS_space': mdpis_space,
        'E' : E,
        'pD0D0' : pdndn,
        'pD0D0S' : pdds,
        'mD0pi_below': mdpi_below,
        'mD0pi_above': mdpi_above,
        'mD0piS_below': mdpiS_below,
        'mD0piS_above': mdpiS_above,
    })

    for key, value in observs.items():
        print(f'{key}:\n{value}')

    # make_observs_plot(observs)


def dndnpi_plot(ax, observs, elo=-2, ehi=15, key='DDpi'):
    ax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(elo, ehi))
    ax.plot(observs['E'], observs[key] / observs[key].max(), label=r'$D^0D^0\pi^+$')


def dndnpis_plot(ax, observs, elo=-2, ehi=15):
    dndnpi_plot(ax, observs, elo=elo, ehi=ehi, key='DDpiS')


def pdd_plot(ax, observs, pddhi, key='pD0D0'):
    ax.set(xlabel=r'$p(DD)$ (MeV)', ylim=(0, 1.01), xlim=(0, pddhi))
    ax.plot(observs[f'{key}space'], observs[key] / observs[key].max(), label=r'$D^0D^0$')


def pdds_plot(ax, observs, pddhi):
    pdd_plot(ax, observs, pddhi, key='pD0D0S')


def dpi_plot(ax, obervs, pdpihi, key='mD0pi'):
    ax.set(xlabel=r'$m(D^0\pi^+)$ (MeV)', ylim=(0, 1.01), xlim=(0, pdpihi))
    mD0pi = obervs[f'{key}_above'] + obervs[f'{key}_below']
    ymax = mD0pi.max()
    ax.plot(obervs[f'{key}_space'], mD0pi / ymax, label=r'$D^0\pi^+$ high')
    ax.plot(obervs[f'{key}_space'], obervs[f'{key}_above'] / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    ax.plot(obervs[f'{key}_space'], obervs[f'{key}_below'] / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')

def dpis_plot(ax, obervs, pdpihi):
    dpi_plot(ax, obervs, pdpihi, key='mD0piS')


def make_observs_plot(observs):
    _, ax = plt.subplots(2, 3, figsize=(18,12))
    for a in ax.ravel():
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    dndnpi_plot(ax[0,0], observs)  # Energy w/o smearing
    dndnpis_plot(ax[1,0], observs)  # Energy w/ smearing
    pdd_plot(ax[0,1], observs, pddhi=150)  # p(DD) w/o smearing
    pdds_plot(ax[1,1], observs, pddhi=150)  # p(DD) w/ smearing
    dpi_plot(ax[0,2], observs, pdpihi=150)  # p(Dpi) w/o smearing
    dpis_plot(ax[1,2], observs, pdpihi=150)  # p(Dpi) w/ smearing

    for a in ax.ravel():
        a.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig('plots/obs.png')
    plt.savefig('plots/obs.pdf')
    plt.show()

if __name__ == '__main__':
    # try:
        elo, ehi = list(map(float, sys.argv[1:]))
        run(elo, ehi)
    # except ValueError:
        # print('Usage: ./observables [E low] [E high]')
