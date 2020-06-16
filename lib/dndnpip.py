""" [D*+ -> D0 pi+] D0 """

import sys
import numpy as np
import matplotlib.pyplot as plt

from .params import gs, gt, mdn, mdn, mpip, mdstp, gamma_star_p, DalitzNBins
from .params import phiins, include_dstndp, include_dd_swave, Rin, g1, g2, norm_swave
from .dalitzphsp import DalitzPhsp
from .lineshape_hanhart import TMtx, RelativisticBreitWigner, MagSq

VERB = False

class DnDnPip(DalitzPhsp):
    """ """
    def __init__(self, gs, gt, E, channels=[include_dstndp, include_dd_swave]):
        super(DnDnPip, self).__init__(E + TMtx.thr, mdn, mdn, mpip)
        self.tmtx = TMtx(gs, gt)
        self.setE(E)
        self.bwdstp = lambda s: RelativisticBreitWigner(s, mdstp, gamma_star_p)
        self.a1 = self.dstn_ampl if channels[0] else lambda x,y: 0
        self.a2 = self.inelastic if channels[1] else lambda: 0

    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t = np.sum(tmtx[0])
        self.tin = Rin * (g1*self.t + g2*np.sum(tmtx[1]))
        self.setM(E + TMtx.thr)
        if VERB:
            print('##### DDPi: E {:.3f} MeV #####'.format(E*10**3))
            print('  mX:  {:.3f} MeV'.format(self.mo*10**3))
            print('   t:  {:.3f}'.format(self.t))

    def inelastic(self):
        return norm_swave * np.exp(1.j*phiins)*self.tin

    def dstn_ampl(self, mdd, md1pi):
        md2pi = self.mZsq(mdd, md1pi)
        return self.t * (self.bwdstp(md1pi) + self.bwdstp(md2pi))

    def calc(self, mdd, md1pi):
        return 2*mpip*self.KineC(mdd) * MagSq(self.a1(mdd, md1pi) + self.a2())
    
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

    def spec(self, E=None, b1=1000, b2=1000, grid=None, high=None):
        if E is not None:
            self.setE(E)
        if grid is None:
            (mdd, md1pi), ds = self.mgridABAC(b1, b2)
        else:
            mdd, md1pi = grid
            ds = (mdd[0,1]-mdd[0,0])*(md1pi[1,0]-md1pi[0,0])
        md2pi = self.mZsq(mdd, md1pi)
        z = self(mdd, md1pi)[0]
        if high is True:
            z[md1pi<md2pi] = 0
        elif high is False:
            z[md1pi>md2pi] = 0
        return ds * z

    def integral(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid))

    def mddspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid), axis=0)

    def mdpihspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid, high=True), axis=1)

    def mdpilspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid, high=False), axis=1)

def tdd(mdd):
    """ Kinetic energy of D0D0 in their frame """
    return mdd - 2.*mdn

def dpi_dpi_plot(ax, pdf, logplot=True):
    (md1pi, md2pi), _ = pdf.mgridACBC(512)
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
    ax.contourf(md1pi, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^0)$', xlabel=r'$m(D^0\pi^+)$')
    ax.contour(md1pi, mdd, phsp, levels=1)

def dd_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAB(nbins)
    mdd = pdf.mddspec(b1=nbins, b2=nbins)
    if sqrt:
        mdd = mdd*2*np.sqrt(bins)
        bins = tdd(np.sqrt(bins))*10**3
        lbl = r'$E(D^0D^0)$, MeV'
    else:
        lbl = r'$m^2(D^0D^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdd)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdd)

def dpi_lo_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdpi = pdf.mdpilspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpi *= 2*bins
        lbl = r'$m(D^0\pi^+)$ low'
    else:
        lbl = r'$m^2(D^0\pi^+)$ low'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

def dpi_hi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdpi = pdf.mdpihspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpi *= 2*bins
        lbl = r'$m(D^0\pi^+)$ high'
    else:
        lbl = r'$m^2(D^0\pi^+)$ high'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

def main():
    """ Unit test """
    import sys
    E = float(sys.argv[1]) * 10**-3

    pdf = DnDnPip(gs, gt, E)
    _, axs = plt.subplots(1, 4, figsize=(16,4))

    # dpi_dpi_plot(axs[0,0], pdf)
    # dd_dpi_plot(axs[1,0], pdf)
    # dd_plot(axs[0,1], pdf, False)
    # dd_plot(axs[1,1], pdf, True)
    # dpi_lo_plot(axs[0,2], pdf, False)
    # dpi_lo_plot(axs[1,2], pdf, True)
    # dpi_hi_plot(axs[0,3], pdf, False)
    # dpi_hi_plot(axs[1,3], pdf, True)

    dpi_dpi_plot(axs[0], pdf)
    dd_plot(     axs[1], pdf, True)
    dpi_lo_plot( axs[2], pdf, True)
    dpi_hi_plot( axs[3], pdf, True)

    plt.tight_layout()

    plt.savefig(f'plots/dp_dndnpip_{E*10**3:.1f}.png')
    plt.savefig(f'plots/dp_dndnpip_{E*10**3:.1f}.pdf')

    plt.show()

if __name__ == '__main__':
    main()
