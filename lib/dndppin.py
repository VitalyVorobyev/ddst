""" [D*+ -> D+ pi0] D0 and [D*0 -> D0 pi0] D+ """

import numpy as np
import matplotlib.pyplot as plt

from params import *
from dalitzphsp import DalitzPhsp
from lineshape_hanhart import TMtx, RelativisticBreitWigner, MagSq

VERB=False

class DnDpPin(DalitzPhsp):
    """ """
    def __init__(self, gs, gt, E, channels=[include_dstndp, include_dstndn], interf=interf_dndstp_dpdstn):
        super(DnDpPin, self).__init__(E + TMtx.thr, mdn, mdp, mpin)
        self.tmtx = TMtx(gs, gt)
        self.setE(E)
        self.bwdstp = lambda s: RelativisticBreitWigner(s, mdstp, gamma_star_p)  # * np.sqrt(br_dstp_dppin)
        self.bwdstn = lambda s: RelativisticBreitWigner(s, mdstn, gamma_star_n)  # * np.sqrt(br_dstn_dnpin)
        self.a1 = self.ampl1 if channels[0] else lambda x: 0
        self.a2 = self.ampl2 if channels[1] else lambda x: 0
        self.pdf = self.wint if interf else self.woint

    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t1 = np.sum(tmtx[0])  # D0 D*+
        self.t2 = np.sum(tmtx[1])  # D+ D*0
        self.setM(E + TMtx.thr)
        if VERB:
            print('##### DDPi: E {:.3f} MeV #####'.format(E*10**3))
            print('  mX:  {:.3f} MeV'.format(self.mo*10**3))
            print('  t1:  {:.3f}'.format(self.t1))
            print('  t2:  {:.3f}'.format(self.t2))

    def wint(self, a1, a2):
        return MagSq(a1-a2)

    def woint(self, a1, a2):
        return MagSq(a1)+MagSq(a2)

    def ampl1(self, mdppi):
        """ D0 D*+ amplitude """
        return self.t1 * self.bwdstp(mdppi)

    def ampl2(self, mdnpi):
        """ D+ D*0 amplitude"""
        return self.t2 * self.bwdstn(mdnpi)

    def calc(self, mdd, mdnpi):
        return self.KineC(mdd) * self.pdf(self.a1(self.mZsq(mdd, mdnpi)), self.a2(mdnpi))

    def __call__(self, mdd, mdnpi):
        mask = self.inPhspABAC(mdd, mdnpi)
        mddv, mdnpiv = [x if isinstance(x, float) else x[mask] for x in [mdd, mdnpi]]
        r = self.calc(mddv, mdnpiv)

        if isinstance(r, float):
            result = r
        else:
            result = np.zeros(mask.shape, dtype=float)
            result[mask] = r
        return (result, mask)

    def spec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        if E is not None:
            self.setE(E)
        if grid is None:
            (mdd, mdnpi), ds = self.mgridABAC(b1, b2)
        else:
            mdd, mdnpi = grid
            ds = (mdd[0,1]-mdd[0,0])*(mdnpi[1,0]-mdnpi[0,0])
        return ds * self(mdd, mdnpi)[0]

    def integral(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid))

    def mddspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid), axis=0)

    def mdnpispec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid), axis=1)

    def mdppispec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        if E is not None:
            self.setE(E)
        if grid is None:
            (mdnpi, mdppi), ds = self.mgridACBC(b1, b2)
        else:
            mdnpi, mdppi = grid
            ds = (mdnpi[0,1]-mdnpi[0,0])*(mdppi[1,0]-mdppi[0,0])
        return ds * np.sum(self(self.mZsq(mdnpi, mdppi), mdnpi)[0], axis=1)

def tdd(mdd):
    """ Kinetic energy of D0D+ in their frame """
    return mdd - mdp - mdn

def dpi_dpi_plot(ax, pdf, logplot=True):
    (mdnpi, mdppi), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(mdnpi, mdppi)
    z, mask = pdf(mdd, mdnpi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnpi, mdppi, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^+\pi^0)$', xlabel=r'$m(D^0\pi^0)$')
    ax.contour(mdnpi, mdppi, phsp, levels=1)

def dd_dpi_plot(ax, pdf, logplot=True):
    (mdd, mdnpi), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, mdnpi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnpi, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^+)$', xlabel=r'$m(D^0\pi^0)$')
    ax.contour(mdnpi, mdd, phsp, levels=1)

def dd_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAB(nbins)
    mdd = pdf.mddspec(b1=nbins, b2=nbins)
    if sqrt:
        mdd = mdd*2*np.sqrt(bins)
        bins = tdd(np.sqrt(bins))*10**3
        lbl = r'$E(D^0D^+)$, MeV'
    else:
        lbl = r'$m^2(D^0D^+)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdd)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdd)

def dnpi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdnpi = pdf.mdnpispec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdnpi *= 2*bins
        lbl = r'$m(D^0\pi^0)$'
    else:
        lbl = r'$m^2(D^0\pi^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdnpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdnpi)

def dppi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdppi = pdf.mdppispec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdppi *= 2*bins
        lbl = r'$m(D^+\pi^0)$'
    else:
        lbl = r'$m^2(D^+\pi^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdppi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdppi)

def main():
    """ Unit test """
    import sys
    E = float(sys.argv[1]) * 10**-3

    pdf = DnDpPin(gs, gt, E)
    fig, axs = plt.subplots(2, 4, figsize=(16,8))

    dpi_dpi_plot(axs[0,0], pdf)
    dd_dpi_plot(axs[1,0], pdf)
    dd_plot(axs[0,1], pdf, False)
    dd_plot(axs[1,1], pdf, True)
    dnpi_plot(axs[0,2], pdf, False)
    dnpi_plot(axs[1,2], pdf, True)
    dppi_plot(axs[0,3], pdf, False)
    dppi_plot(axs[1,3], pdf, True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
