#! /usr/bin/env python
""" """

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

from lib.params import gs, gt
from lib.params import br_dstn_dngam, br_dstn_dnpin
from lib.params import br_dstp_dnpip, br_dstp_dppin
from lib.dndnpip import DnDnPip
from lib.dndppin import DnDpPin
from lib.dndpgam import DnDpGam

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
        print('E {:.3f} MeV'.format(energy*10**3))
        for i, p, g in zip(I, pdf, grid):
            p.setE(energy)
            p.t1, p.t2, p.t = 1, 1, 1
            i[idx] = p.integral(grid=g)

    gam_over_pi0 = I[3][-1] / I[2][-1]
    scale = (br_dstn_dngam / br_dstn_dnpin) / gam_over_pi0
    print(f'gam / pi0 = {gam_over_pi0:.3f}')
    print(f'conmare to {br_dstn_dngam / br_dstn_dnpin:.3f}')
    print(f'scaling: {scale:.3f}')

    E *= 10**3
    plt.figure(figsize=(8,6))
    plt.plot(E, I[0] / I[1], label=r'$D^{*+}\to D^0\pi^+$ / $D^{*+}\to D^+\pi^0$')
    plt.plot([E[0], E[-1]], [[br_dstp_dnpip / br_dstp_dppin] for _ in range(2)], '--',
        label=r'BF($D^{*+}\to D^0\pi^+$) / BF($D^{*+}\to D^+\pi^0$)')
    plt.plot(E, I[3] / I[2], label=r'$D^{*0}\to D^0\gamma$ / $D^{*0}\to D^0\pi^0$')
    plt.plot([E[0], E[-1]], [[br_dstn_dngam / br_dstn_dnpin] for _ in range(2)], '--',
        label=r'BF($D^{*0}\to D^0\gamma$) / BF($D^{*0}\to D^0\pi^0$)')
    plt.plot(E, I[2] / I[0], label=r'$D^{*0}\to D^0\pi^0$ / $D^{*+}\to D^0\pi^+$')
    plt.xlim(E[0], E[-1])
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.tight_layout()
    plt.grid()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/gammanorm.{ext}')

    plt.figure()
    plt.plot(E, I[2], label=r'$\Gamma(D^{*0}\to D^0\pi^0)$')
    plt.plot(E, I[3], label=r'$\Gamma(D^{*0}\to D^0\gamma)$')
    plt.xlim(E[0], E[-1])
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$E$ (MeV)', fontsize=18)
    plt.tight_layout()
    plt.grid()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/gammanorm_pi0_gam.{ext}')

    plt.show()

if __name__ == '__main__':
    elo, ehi = [float(x) for x in sys.argv[1:]]
    getespec(elo, ehi)

