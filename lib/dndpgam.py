""" [D*0 -> D0 gamma] D+ """

import numpy as np
import matplotlib.pyplot as plt

from .params import gs, gt, mdn, mdp, GammaScale
from .dalitzphsp import DalitzPhsp
from .lineshape_hanhart import TMtx, RbwDstn, MagSq

VERB=False

class DnDpGam(DalitzPhsp):
    """ """
    def __init__(self, gs, gt, E):
        super(DnDpGam, self).__init__(E + TMtx.thr, mdn, mdp, 0)
        self.tmtx = TMtx(gs, gt)
        self.bwdstn = lambda s: RbwDstn(s)
        self.setE(E)

    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t2 = np.sum(tmtx[1])  # D+ D*0
        self.setM(E + TMtx.thr)
        if VERB:
            print('##### DDGa: E {:.3f} MeV #####'.format(E*10**3))
            print('   s:  {:.3f} MeV'.format(self.mo*10**3))
            print('  t2:  {:.3f}'.format(self.t2))

    def calc(self, mdd, mdnga):
        return self.KineC(mdd)**2 * MagSq(self.t2 * self.bwdstn(mdnga)) * GammaScale

    def __call__(self, mdd, mdnga):
        mask = self.inPhspABAC(mdd, mdnga)
        mddv, mdngav = [x if isinstance(x, float) else x[mask] for x in [mdd, mdnga]]
        r = self.calc(mddv, mdngav)

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
            (mdd, mdnga), ds = self.mgridABAC(b1, b2)
        else:
            mdd, mdnga = grid
            ds = (mdd[0,1]-mdd[0,0])*(mdnga[1,0]-mdnga[0,0])
        return ds * self(mdd, mdnga)[0]

    def integral(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid))

    def mddspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid), axis=0)

    def mdngaspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        return np.sum(self.spec(E, b1, b2, grid), axis=1)

    def mdpgaspec(self, E=None, b1=1000, b2=1000, grid=None):
        """ """
        if E is not None:
            self.setE(E)
        if grid is not None:
            mdnga, mdpga = grid
            ds = (mdnga[0,1]-mdnga[0,0])*(mdpga[1,0]-mdpga[0,0])
        else:
            (mdnga, mdpga), ds = self.mgridACBC(b1, b2)
        return ds * np.sum(self(self.mZsq(mdnga, mdpga), mdnga)[0], axis=1)

def tdd(mdd):
    """ Kinetic energy of D0D- in their frame """
    return mdd - mdp - mdn

def dga_dga_plot(ax, pdf, logplot=True):
    (mdnga, mdpga), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(mdnga, mdpga)
    z, mask = pdf(mdd, mdnga)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnga, mdpga, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^+\gamma)$', xlabel=r'$m(D^0\gamma)$')
    ax.contour(mdnga, mdpga, phsp, levels=1)

def dd_dga_plot(ax, pdf, logplot=True):
    (mdd, mdnga), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, mdnga)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnga, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^+)$', xlabel=r'$m(D^0\gamma)$')
    ax.contour(mdnga, mdd, phsp, levels=1)

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

def dnga_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdnga = pdf.mdngaspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdnga *= 2*bins
        lbl = r'$m(D^0\gamma)$'
    else:
        lbl = r'$m^2(D^0\gamma)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdnga)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdnga)

def dpga_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdpga = pdf.mdpgaspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpga *= 2*bins
        lbl = r'$m(D^+\gamma)$'
    else:
        lbl = r'$m^2(D^+\gamma)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpga)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpga)

def main():
    """ Unit test """
    import sys
    gs = (30 + 0.5j) * 10**-3
    gt = (-30 + 0.5j) * 10**-3
    E = float(sys.argv[1]) * 10**-3

    pdf = DnDpGam(gs, gt, E)
    fig, axs = plt.subplots(2, 4, figsize=(16,8))

    dga_dga_plot(axs[0,0], pdf)
    dd_dga_plot(axs[1,0], pdf)
    dd_plot(axs[0,1], pdf, False)
    dd_plot(axs[1,1], pdf, True)
    dnga_plot(axs[0,2], pdf, False)
    dnga_plot(axs[1,2], pdf, True)
    dpga_plot(axs[0,3], pdf, False)
    dpga_plot(axs[1,3], pdf, True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
