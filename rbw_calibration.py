""" """

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

from lib.params import gamma_star_n_dngam, gamma_star_n_dnpin, gamma_star_n
from lib.params import mdstn, mdn, mpin, GammaScale
from lib.lineshape import RbwDstn, MagSq
from lib.dalitzphsp import two_body_momentum

mdnSq = mdn**2
mpinSq = mpin**2
mdstnSq = mdstn**2

def pdf_dstn_dn_pin(s):
    """ """
    return two_body_momentum(s, mdnSq, mpinSq)**3 * MagSq(RbwDstn(s))

def pdf_dstn_dn_gam(s):
    """ """
    return two_body_momentum(s, mdnSq, 0)**1 * MagSq(RbwDstn(s))

def run(nsigma=10):
    elo = mdstn - nsigma*gamma_star_n
    ehi = mdstn + nsigma*gamma_star_n
    E = np.linspace(elo, ehi, 512)
    dE = E[1] - E[0]

    Esq = E**2
    ppin = pdf_dstn_dn_pin(Esq)
    pgam = pdf_dstn_dn_gam(Esq)

    Ipin = np.sum(ppin) * dE
    Igam = np.sum(pgam) * dE

    # Ipin / gamma_star_n_dnpin = scale * Igam / gamma_star_n_dngam
    gam_scale = Ipin / gamma_star_n_dnpin * gamma_star_n_dngam / Igam
    print(f'gam fact {gam_scale*10**3:.3f}e-3')
    # gamma_star_n_dnpin = Ipin * absolute_factor
    absolute_factor = gamma_star_n_dnpin / Ipin
    print(f'abs fact {absolute_factor*10**3:.3f}e-3')

    ppin *= absolute_factor
    pgam *= absolute_factor * gam_scale

    E = (E - mdstn)*10**6
    plt.figure(figsize=(8,6))
    plt.plot(E, ppin, label=r'$D^0\pi^0$')
    plt.plot(E, pgam, label=r'$D^0\gamma$')
    plt.xlabel(r'$\Delta E$ (keV)', fontsize=16)
    plt.ylim([0, 1.05*max(ppin)])
    plt.xlim([E[0], E[-1]])
    plt.grid()
    plt.legend(loc='best', fontsize=16)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/rbw_norm.{ext}')

    plt.show()

if __name__ == '__main__':
    print(f'pi0 momentum {two_body_momentum(mdstnSq, mdnSq, mpinSq)*10**3:.3f} MeV')
    print(f'gam momentum {two_body_momentum(mdstnSq, mdnSq, 0)*10**3:.3f} MeV')
    run(float(sys.argv[1]))
