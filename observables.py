#! /usr/bin/env python
""" m(D0D0pi+)
    m(D0D0): D0[D*+ -> D0 pi+]
    m(D0pi+)
    m(D0D+): D0[D*+ -> D+ pi0], D+[D*0 -> D0 pi0], D+[D*0 -> D0 gamma]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.resolution import smear_e0, smear_mdpi, smear_pd
from lib.params import gs, gt, mdn, mdp
import lib.vartools as vt
import lib.resolution as res

include_swave = False

def init_observs(elo=-2, ehi=8, nEbins=256, nABbins=256, nACbins=256, nBCbins=256):
    """ """
    E = np.linspace(elo, ehi, nEbins)
    ddpipFull = DnDnPip(gs, gt, E[-1], [True, include_swave])

    ab_space = ddpipFull.linspaceAB(nABbins)
    ac_space = ddpipFull.linspaceAC(nACbins)
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

    data = {
        'DDpi' : np.zeros(nEbins),
        'D0D0' : np.zeros(nABbins),
        'D0pi' : [np.zeros(nACbins), np.zeros(nACbins)],
        'DDpiS' : np.zeros(nACbins),
    }

    def update_d0d0(idx, mab, mac):
        data['D0D0'] += mab

    def update_ddpi(idx, mab, mac):
        data['DDpi'][idx] = mab.sum() * delta_ab

    def update_d0pi(idx, mab, mac):
        if E[idx] > 0:
            data['D0pi'][1] += mac
        else:
            data['D0pi'][0] += mac

    def update_ddpis(idx, mab, mac):
        pd_sp, pd_pdf = vt.transform_distribution(ab_space, mab, vt.mddsq_to_pd, nABbins)
        data['DDpiS'] += smear_e0(E[idx], E, pd_sp, pd_pdf)

    obs = [
        ['DDpi', data['DDpi'], update_ddpi],
        ['D0D0', data['D0D0'], update_d0d0],
        ['D0pi', data['D0pi'], update_d0pi],
        ['DDpiS', data['DDpiS'], update_ddpis],
        ['E', E, None],
    ]

    return (E, processEnergy, obs, ab_space, ac_space)


def run(elo=-2, ehi=8):
    """ """
    nbins = 512
    E, processEnergy, observs, ab_space, ac_space = init_observs(
        elo=elo, ehi=ehi, nEbins=nbins, nABbins=nbins, nACbins=nbins, nBCbins=nbins)

    for idx, energy in enumerate(E):
        print(f'{idx:>3}/{E.size}: E {energy:.3f} MeV')
        mab, mac = processEnergy(energy)
        for _, _, fcn in observs:
            if fcn is not None:
                fcn(idx, mab, mac)

    spline_dots = 1024
    observs = {label: data for label, data, _ in  observs}
    
    mdpi_space, mdpi_below = vt.transform_distribution(ac_space, observs['D0pi'][0], vt.msq_to_m, spline_dots)
    _         , mdpi_above = vt.transform_distribution(ac_space, observs['D0pi'][1], vt.msq_to_m, spline_dots)

    mdpis_space, mdpiS_below = smear_mdpi(mdpi_space, mdpi_below, spline_dots)
    _          , mdpiS_above = smear_mdpi(mdpi_space, mdpi_above, spline_dots)

    pd_space, pd = vt.transform_distribution(ab_space, observs['D0D0'], vt.mddsq_to_pd, spline_dots)
    pds_space, pds = smear_pd(pd_space, pd, dots=spline_dots)

    observs.update({
        'pD0_space' : pd_space,
        'pD0' : pd,
        'pD0S_space' : pds_space,
        'pD0S' : pds,
        'mD0pi_space': mdpi_space,
        'mD0pi_below': mdpi_below,
        'mD0pi_above': mdpi_above,
        'mD0piS_space': mdpis_space,
        'mD0piS_below': mdpiS_below,
        'mD0piS_above': mdpiS_above,
    })

    observ_plots(observs)


def dndnpi_plot(ax, observs, elo=-2, ehi=15, key='DDpi'):
    E = observs['E']
    ax.set(xlabel=r'$E$ (MeV)', ylim=(0, 1.01), xlim=(max(elo, E[0]), min(ehi, E[-1])))
    ax.plot(observs['E'], observs[key] / observs[key].max(), label=r'$D^0D^0\pi^+$')


def dndnpis_plot(ax, observs, elo=-2, ehi=15):
    dndnpi_plot(ax, observs, elo=elo, ehi=ehi, key='DDpiS')


def pdd_plot(ax, observs, pddhi, key='pD0', setlabel=False):
    ax.set(ylim=(0, 1.01), xlim=(0, pddhi))
    if setlabel:
        ax.set(xlabel=r'$p(D)$ (MeV)')
    ax.plot(observs[f'{key}_space'], observs[key].ravel() / observs[key].max(), label=r'$D^0D^0$')

def pdds_plot(ax, observs, pddhi):
    pdd_plot(ax, observs, pddhi, key='pD0S', setlabel=True)


def dpi_plot(ax, obervs, pdpihi, key='mD0pi'):
    ax.set(xlabel=r'$m(D^0\pi^+)$ (MeV)', ylim=(0, 1.01), xlim=(2006, 2016))
    mD0pi = obervs[f'{key}_above'] + obervs[f'{key}_below']
    ymax = mD0pi.max()
    ax.plot(obervs[f'{key}_space'], mD0pi.ravel() / ymax, label=r'$D^0\pi^+$ high')
    ax.plot(obervs[f'{key}_space'], obervs[f'{key}_above'].ravel() / ymax, '--', label=r'$D^0\pi^+$ high, $E>0$')
    ax.plot(obervs[f'{key}_space'], obervs[f'{key}_below'].ravel() / ymax, ':',  label=r'$D^0\pi^+$ high, $E<0$')


def dpis_plot(ax, obervs, pdpihi):
    dpi_plot(ax, obervs, pdpihi, key='mD0piS')


def observ_plots(observs):
    _, ax = plt.subplots(2, 3, figsize=(18,12), sharex='col')
    for a in ax.ravel():
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    dndnpi_plot(ax[0,0], observs)  # Energy w/o smearing
    dndnpis_plot(ax[1,0], observs)  # Energy w/ smearing
    pdd_plot(ax[0,1], observs, pddhi=300)  # p(DD) w/o smearing
    pdds_plot(ax[1,1], observs, pddhi=300)  # p(DD) w/ smearing
    dpi_plot(ax[0,2], observs, pdpihi=150)  # p(Dpi) w/o smearing
    dpis_plot(ax[1,2], observs, pdpihi=150)  # p(Dpi) w/ smearing

    for a in ax.ravel():
        a.legend(fontsize=16)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/obs.{ext}')
    plt.show()

if __name__ == '__main__':
    try:
        elo, ehi = list(map(float, sys.argv[1:]))
        run(elo, ehi)
    except ValueError:
        print('Usage: ./observables [E low] [E high]')
