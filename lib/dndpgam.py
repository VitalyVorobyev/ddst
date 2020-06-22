""" [D*0 -> D0 gamma] D+ """

import numpy as np
import matplotlib.pyplot as plt

from .params import gs, gt, mdn, mdp, GammaScale
from .dalitzphsp import DalitzPhsp
from .lineshape import TMtx, RbwDstn, MagSq

class DnDpGam(DalitzPhsp):
    """ The [X -> D+ [D*0 -> D0 gamma]] decay amplitude """

    verb=False
    def __init__(self, gs, gt, E):
        super(DnDpGam, self).__init__(E + TMtx.thr, mdn, mdp, 0)
        self.tmtx = TMtx(gs, gt)
        self.bwdstn = lambda s: RbwDstn(s)
        self.setE(E)

    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t2 = np.sum(tmtx[1])  # D+ D*0
        self.setM(E + TMtx.thr)
        if self.verb:
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
