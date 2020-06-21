""" [D*+ -> D+ pi0] D0 and [D*0 -> D0 pi0] D+ """

import numpy as np
import matplotlib.pyplot as plt

from .params import gs, gt, mdn, mdp, mpin, mdstp, gamma_star_p
from .params import Rin, g1, g2, phiinp
from .params import norm_pwave, alpha_pwave, DalitzNBins
from .params import include_dstpdn, include_dstndp, include_dd_pwave, interf_dndstp_dpdstn
from .dalitzphsp import DalitzPhsp
from .lineshape_hanhart import TMtx, RelativisticBreitWigner, MagSq, RbwDstn

class DnDpPin(DalitzPhsp):
    """ The [X -> D0 D+ pi0] decay amplitude """

    verb=False

    def __init__(self, gs, gt, E, channels=[include_dstpdn, include_dstndp, include_dd_pwave], interf=interf_dndstp_dpdstn):
        super(DnDpPin, self).__init__(E + TMtx.thr, mdn, mdp, mpin)
        self.tmtx = TMtx(gs, gt)
        self.setE(E)
        self.alpha = alpha_pwave
        self.bwdstp = lambda s: RelativisticBreitWigner(s, mdstp, gamma_star_p)
        self.bwdstn = lambda s: RbwDstn(s)
        self.a1 = self.ampl1     if channels[0] else lambda x: 0
        self.a2 = self.ampl2     if channels[1] else lambda x: 0
        self.a3 = self.inelastic if channels[2] else lambda x,y: 0
        self.pdf = self.wint     if interf else self.woint


    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t1 = np.sum(tmtx[0])  # D0 D*+
        self.t2 = np.sum(tmtx[1])  # D+ D*0
        self.tin = Rin * (g1*self.t1 - g2*self.t2)
        self.setM(E + TMtx.thr)
        if self.verb:
            print('##### DDPi: E {:.3f} MeV #####'.format(E*10**3))
            print('  mX:  {:.3f} MeV'.format(self.mo*10**3))
            print('  t1:  {:.3f}'.format(self.t1))
            print('  t2:  {:.3f}'.format(self.t2))


    def wint(self, a1, a2, a3):
        """ """
        return MagSq(a1+a2+a3)


    def woint(self, a1, a2, a3):
        """ """
        return MagSq(a1)+MagSq(a2)+MagSq(a3)


    def ampl1(self, mdppi):
        """ D0 D*+ amplitude """
        return self.t1 * self.bwdstp(mdppi)


    def ampl2(self, mdnpi):
        """ D+ D*0 amplitude"""
        return self.t2 * self.bwdstn(mdnpi)


    def inelastic(self, mdd, mdppi):
        """ inelastic channel from T-matrix """
        return np.exp(1.j*phiinp) * self.tin * self.dbl_pBpC_AB(mdd, mdppi)


    def pwave(self, mdd, mdppi):
        """ (D0D+) P-wave amplitude """
        return (self.dbl_pBpC_AB(mdd, mdppi) + self.dbl_pBpC_AB(mdd, self.mZsq(mdd, mdppi))) * np.exp(-self.alpha * mdd)


    def calc(self, mdd, mdnpi):
        mdppi = self.mZsq(mdd, mdnpi)
        return 2*mpin*self.KineC(mdd) * self.pdf(
            self.a1(mdppi), self.a2(mdnpi), norm_pwave*self.a3(mdd, mdppi))


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
