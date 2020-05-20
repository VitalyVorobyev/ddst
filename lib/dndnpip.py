""" [D*+ -> D0 pi+] D0 """

import numpy as np
import matplotlib.pyplot as plt

from params import *
from dalitzphsp import DalitzPhsp
from lineshape_hanhart import TMtx, RelativisticBreitWigner, MagSq

class DnDnPip(DalitzPhsp):
    """ """
    def __init__(self, gs, gt, E):
        super(DnDnPip, self).__init__(E + TMtx.thr, mdn, mdn, mpip)
        self.tmtx = TMtx(gs, gt)
        self.setE(E)
        self.bwdstp = lambda s: RelativisticBreitWigner(s, mdstp, gamma_star_p)

    def setE(self, E):
        self.t = np.sum(self.tmtx(E)[0])
        self.setM(E + TMtx.thr)
        self.bwden = TMtx.mu * (2.*E + 1j*gamma_star_p)
        print('##### DDPi: E {:.3f} MeV #####'.format(E*10**3))
        # print('  mX:  {:.3f} MeV'.format(self.mo*10**3))
        # print('   t:  {:.3f}'.format(self.t))

    def calc(self, mdd, md1pi):
        md2pi = self.mZsq(mdd, md1pi)
        ampl = self.t * (self.bwdstp(md1pi) + self.bwdstp(md2pi))
        return self.KineC(mdd) * MagSq(ampl)

    def __call__(self, mdd, md1pi):
        mask = self.inPhspABAC(mdd, md1pi)
        mddv, md1piv = [x if isinstance(x, float) else x[mask] for x in [mdd, md1pi]]
        r = self.calc(mddv, md1piv)

        if isinstance(r, float):
            result = r
        else:
            result = np.zeros(mask.shape, dtype=float)
            result[mask] = r
        return (result, mask)

    def integral(self, E, b1=1000, b2=1000):
        """ """
        self.setE(E)
        (mdd, mdpi), ds = self.mgridABAC(b1, b2)
        return ds * np.sum(self(mdd, mdpi))

    def spec(self, E=None, b1=1000, b2=1000):
        if E is not None:
            self.setE(E)
        (mdd, md1pi), ds = self.mgridABAC(b1, b2)
        md2pi = self.mZsq(mdd, md1pi)
        z = self(mdd, md1pi)[0]
        z[md1pi<md2pi] = 0
        return (z, ds)

    def mddspec(self, E=None, b1=1000, b2=1000):
        """ """
        if E is not None:
            self.setE(E)
        z, ds = self.spec(E, b1, b2)
        return ds * np.sum(z, axis=0)

    def mdpispec(self, E=None, b1=1000, b2=1000):
        """ """
        if E is not None:
            self.setE(E)
        z, ds = self.spec(E, b1, b2)
        return ds * np.sum(z, axis=1)

    def mddbins(self, b1=1000):
        return np.linspace(self.mABsqRange[0], self.mABsqRange[1], b1)

    def mdpibins(self, b2=1000):
        return np.linspace(self.mACsqRange[0], self.mACsqRange[1], b2)

def tdd(mdd):
    """ Kinetic energy of D0D0 in their frame """
    return mdd - 2.*mdn

def dpi_dpi_plot(ax, pdf, logplot=True):
    (md1pi, md2pi), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(md1pi, md2pi)
    z, mask = pdf(mdd, md1pi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(md1pi, md2pi, z, cmap=None, levels=100)
    ax.set(xlabel=r'$m(D^0_{(1)}\pi^+)$', ylabel=r'$m(D^0_{(2)}\pi^+)$')
    ax.contour(md1pi, md2pi, phsp, levels=1)

def dd_dpi_plot(ax, pdf, logplot=True):
    (mdd, md1pi), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, md1pi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdd, md1pi, z, cmap=None, levels=100)
    ax.set(xlabel=r'$m(D^0D^0)$', ylabel=r'$m(D^0\pi^+)$')
    ax.contour(mdd, md1pi, phsp, levels=1)

def dd_plot(ax, pdf, sqrt=True):
    bins = pdf.mddbins()
    mdd = pdf.mddspec()
    if sqrt:
        mdd = mdd*2*np.sqrt(bins)
        bins = tdd(np.sqrt(bins))*10**3
        lbl = r'$E(D^0D^0)$, MeV'
    else:
        lbl = r'$m^2(D^0D^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdd)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdd)

def dpi_plot(ax, pdf, sqrt=True):
    bins = pdf.mdpibins()
    mdpi = pdf.mdpispec()
    if sqrt:
        bins = np.sqrt(bins)
        mdpi *= 2*bins
        lbl = r'$m(D^0\pi^0)$'
    else:
        lbl = r'$m^2(D^0\pi^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

def main():
    """ Unit test """
    import sys
    E = float(sys.argv[1]) * 10**-3

    pdf = DnDnPip(gs, gt, E)
    fig, axs = plt.subplots(2, 3, figsize=(12,8))

    dpi_dpi_plot(axs[0,0], pdf)
    dd_dpi_plot(axs[1,0], pdf)
    dd_plot(axs[0,1], pdf, False)
    dd_plot(axs[1,1], pdf, True)
    dpi_plot(axs[0,2], pdf, False)
    dpi_plot(axs[1,2], pdf, True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
