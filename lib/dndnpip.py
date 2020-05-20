""" [D*+ -> D0 pi+] D0 """

import numpy as np
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

    def integral(self, E, b1=500, b2=500):
        """ """
        self.setE(E)
        (mdd, mdpi), ds = self.mgridABAC(b1, b2)
        return ds * np.sum(self(mdd, mdpi))

    def spec(self, E, b1, b2):
        self.setE(E)
        (mdd, md1pi), ds = self.mgridABAC(b1, b2)
        md2pi = self.mZsq(mdd, md1pi)
        z = self(mdd, md1pi)[0]
        z[md1pi<md2pi] = 0
        return (z, ds)

    def mddspec(self, E, b1=500, b2=500):
        """ """
        z, ds = self.spec(E, b1, b2)
        return ds * np.sum(z, axis=0)

    def mdpispec(self, E, b1=500, b2=500):
        """ """
        z, ds = self.spec(E, b1, b2)
        return ds * np.sum(z, axis=1)

    def mddbins(self, b1):
        return np.linspace(self.mABsqRange[0], self.mABsqRange[1], b1)

    def mdpibins(self, b2):
        return np.linspace(self.mACsqRange[0], self.mACsqRange[1], b2)

def main():
    """ Unit test """
    import sys
    # gs = (62.2 + 0.5j) * 10**-3
    # gt = (19.7 + 1.6j) * 10**-3
    gs = (30 + 0.5j) * 10**-3
    gt = (30 + 0.5j) * 10**-3
    E = float(sys.argv[1]) * 10**-3

    pdf = DnDnPip(gs, gt, E)
    # (mdd, mdpi), _ = pdf.mgridABAC(500)
    (md1pi, md2pi), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(md1pi, md2pi)

    import matplotlib.pyplot as plt
    z, mask = pdf(mdd, md1pi)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    plt.figure(figsize=(8,8))
    # plt.contourf(mdd, mdpi, z)
    # plt.contour(mdd, mdpi, phsp, levels=1)
    # plt.contourf(md1pi, md2pi, np.log(z+1.000000001), cmap=None, levels=100)
    plt.contourf(md1pi, md2pi, z, cmap=None, levels=100)
    plt.contour(md1pi, md2pi, phsp, levels=1)
    plt.xlabel(r'$m(D^0_{(1)}\pi^+)$', fontsize=18)
    plt.ylabel(r'$m(D^0_{(2)}\pi^+)$', fontsize=18)
    plt.title(r'$[D^{*+} \to D^0 \pi^+] D^0$' + f'  E={E*1000} MeV', fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
