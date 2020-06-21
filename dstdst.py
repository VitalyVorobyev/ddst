""" A primitive toy study of the (D^*0 D^*+) contribution """

import numpy as np
import matplotlib.pyplot as plt

import phasespace as phsp

from lib.params import mdn, mpip, mpin, mdstp, mdstn

def as_numpy(p):
    """ tensors to np.array """
    return {key: v.numpy() for key, v in p.items()}

def generate(sqrts, N, h='pi0'):
    """ Generates [D*0 -> D^0 \pi^+][D^*0 -> D0 h],
        where h is in {pi0, gamma}, decays
    """
    assert h in ['pi0', 'gamma']
    assert sqrts >= mdstn + mdstp

    pip = phsp.GenParticle('pi+', mpip)
    h = phsp.GenParticle('pi0', mpin) if h == 'pi0' else phsp.GenParticle('gamma', 0)
    dn1 = phsp.GenParticle('D0_D*0', mdn)
    dn2 = phsp.GenParticle('D0_D*+', mdn)
    dstn = phsp.GenParticle('D*0', mdstn).set_children(dn1, h)
    dstp = phsp.GenParticle('D*+', mdstp).set_children(dn2, pip)
    x = phsp.GenParticle('X', sqrts).set_children(dstp, dstn)

    weights, particles =  x.generate(N)
    return (weights.numpy(), as_numpy(particles))

def mass(lv):
    """ Invariant mass of a four-momentum """
    return np.sqrt(lv[:,-1]**2 - np.sum(lv[:,:-1]**2, axis=-1))

def mddpi(p):
    """ m(D0D0pi+) """
    return mass(p['D0_D*0'] + p['D0_D*+'] + p['pi+'])

def mdd(p):
    """ m(D0D0) """
    return mass(p['D0_D*0'] + p['D0_D*+'])

def make_hist(m, w, nbins=250):
    hist, bins = np.histogram(m, bins=nbins, weights=w, density=False)
    errs = np.sqrt(hist)
    bins = 0.5*(bins[1:]+bins[:-1])
    return (bins, hist, errs)

def make_hist2d(x, y, w, nbins=250):
    z, binsx, binsy = np.histogram2d(x, y, bins=nbins, weights=w)
    binsx, binsy = [0.5*(v[1:]+v[:-1]) for v in [binsx, binsy]]
    xmg, ymg = np.meshgrid(binsx, binsy)
    return (xmg, ymg, z)

def main():
    sqrts = mdstn + mdstp + 1.e-3  # 1 MeV above the threshold
    w, p = generate(sqrts, 10**6, 'gamma')

    nbins = 40

    draw_params = {
        'linestyle': 'none',
        'marker': '.',
        'markersize': 6
    }

    mddpi_lbl = r'$\Delta m(D^0D^0\pi^+)$ (MeV)'
    mdd_lbl = r'$\Delta m(D^0D^0)$ (MeV)'

    # m(D0D0pi+) #
    m1 = (mddpi(p) - 2*mdn - mpip)*10**3
    x1, y1, yerr1 = make_hist(m1, w, nbins)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x1, y1, yerr=yerr1, **draw_params)
    plt.xlabel(mddpi_lbl, fontsize=16)
    plt.grid()
    plt.xlim(0, 10)
    plt.ylim(0, 1.05*max(y1))

    for ext in ['pdf', 'png']:
        plt.savefig(f'plots/dstdst_ddpi.{ext}')
    ####

    # m(D0D0) #
    m2 = (mdd(p) - 2*mdn)*10**3
    x2, y2, yerr2 = make_hist(m2, w, nbins)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x2, y2, yerr=yerr2, **draw_params)
    plt.xlabel(mdd_lbl, fontsize=16)
    plt.grid()
    plt.xlim(0, 10)
    plt.ylim(0, 1.05*max(y2))

    for ext in ['pdf', 'png']:
        plt.savefig(f'plots/dstdst_dd.{ext}')
    ####

    # Contours #
    xmg, ymg, z = make_hist2d(m2, m1, w, nbins=nbins)
    plt.figure(figsize=(8, 8))
    plt.contourf(xmg, ymg, z, levels=15)
    plt.xlabel(mdd_lbl, fontsize=16)
    plt.ylabel(mddpi_lbl, fontsize=16)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(f'plots/dstdst_contours.{ext}')
    ####

    plt.show()

if __name__ == '__main__':
    main()
