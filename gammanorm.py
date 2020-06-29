#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.params import gs, gt
from lib.params import br_dstn_dngam, br_dstn_dnpin
from lib.params import br_dstp_dnpip, br_dstp_dppin
from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam
from lib.plots import put_plot_on_axis

def getespec(emin=3, emax=10):
    """ """
    N = 128
    bins=2048

    E = np.linspace(emin, emax, N)*10**-3

    pdf = [
        DnDnPip(gs, gt, E[-1], channels=[True, False]),
        DnDpPin(gs, gt, E[-1], channels=[True, False, False]),
        DnDpPin(gs, gt, E[-1], channels=[False, True, False]),
        DnDpGam(gs, gt, E[-1])
    ]

    grid = [p.mgridABAC(bins, bins)[0] for p in pdf]
    I = [np.zeros(N) for _ in pdf]

    for idx, energy in enumerate(E):
        print(f'E {energy*10**3:.3f} MeV ({idx}/{E.shape[0]})')
        for i, p, g in zip(I, pdf, grid):
            p.setE(energy)
            p.t1, p.t2, p.t, p.tin = 1, 1, 1, 1
            i[idx] = p.integral(grid=g)

    gam_over_pi0 = I[3][-1] / I[2][-1]
    scale = (br_dstn_dngam / br_dstn_dnpin) / gam_over_pi0
    print(f'gam / pi0 = {gam_over_pi0:.3f}')
    print(f'compare to {br_dstn_dngam / br_dstn_dnpin:.3f}')
    print(f'scaling: {scale:.3f}')

    E *= 10**3
    figax1 = plt.subplots(figsize=(8,6))
    put_plot_on_axis(figax=figax1, xlabel=r'$E$ (MeV)', xlim=(E[0], E[-1]),
        saveas='gammanorm',
        data = [
            [
                E, I[0] / I[1],
                r'$D^{*+}\to D^0\pi^+$ / $D^{*+}\to D^+\pi^0$', {}
            ], [
                [E[0], E[-1]],
                [[br_dstp_dnpip / br_dstp_dppin] for _ in range(2)],
                r'BF($D^{*+}\to D^0\pi^+$) / BF($D^{*+}\to D^+\pi^0$)',
                {'linestyle': '--'}
            ], [
                E, I[3] / I[2],
                r'$D^{*0}\to D^0\gamma$ / $D^{*0}\to D^0\pi^0$', {}
            ], [
                [E[0], E[-1]],
                [[br_dstn_dngam / br_dstn_dnpin] for _ in range(2)],
                r'BF($D^{*0}\to D^0\gamma$) / BF($D^{*0}\to D^0\pi^0$)',
                {'linestyle': '--'}
            ], [
                E, I[2] / I[0],
                r'$D^{*0}\to D^0\pi^0$ / $D^{*+}\to D^0\pi^+$', {}
            ]
        ]
    )

    figax2 = plt.subplots(figsize=(8,6))
    put_plot_on_axis(figax=figax2, xlabel=r'$E$ (MeV)', xlim=(E[0], E[-1]),
        saveas='gammanorm_pi0_gam',
        data = [
            [E, I[2], r'$\Gamma(D^{*0}\to D^0\pi^0)$', {}],
            [E, I[3], r'$\Gamma(D^{*0}\to D^0\gamma)$', {}]
        ]
    )

    plt.show()

if __name__ == '__main__':
    try:
        elo, ehi = map(float, sys.argv[1:])
        getespec(elo, ehi)
    except ValueError:
        print('Usage: ./gammanorm.py [E low] [E high]')
