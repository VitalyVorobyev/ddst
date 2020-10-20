""" [D*+ -> D0 pi+] D0 """

import sys
import numpy as np

from .params import gs, gt, mdn, mdn, mpip, mdstp, gamma_star_p, DalitzNBins
from .params import phiins, include_dstndp, include_dd_swave, Rin, g1, g2, norm_swave
from .dalitzphsp import DalitzPhsp, Kibble
from .lineshape import TMtx, RelativisticBreitWigner, MagSq
from . import vartools as vt


class DnDnPip(DalitzPhsp):
    """ The [Tcc -> D0 [D*+ -> D0 pi+]] decay amplitude """

    verb=False

    def __init__(self, gs=gs, gt=gt, E=0, channels=[include_dstndp, include_dd_swave]):
        super(DnDnPip, self).__init__(E + TMtx.thr, mdn, mdn, mpip)
        self.tmtx = TMtx(gs, gt)
        self.setE(E)
        self.bwdstp = lambda s: RelativisticBreitWigner(s, mdstp, gamma_star_p)
        self.a1 = self.dstn_ampl if channels[0] else lambda x,y: 0
        self.a2 = self.inelastic if channels[1] else lambda: 0
        self.isInPhsp = lambda s, md1pisq, mddsq:\
            Kibble(s, md1pisq, mddsq, mdn**2, mdn**2, mpip**2)

    def set_gs_gt(self, gs, gt):
        self.tmtx = TMtx(gs, gt)

    def setE(self, E):
        tmtx = self.tmtx(E)
        self.t = np.sum(tmtx[0])
        self.tin = Rin * (g1*self.t + g2*np.sum(tmtx[1]))
        self.setM(E + TMtx.thr)
        if self.verb:
            print('##### DDPi: E {:.3f} MeV #####'.format(E*10**3))
            print('  mX:  {:.3f} MeV'.format(self.mo*10**3))
            print('   t:  {:.3f}'.format(self.t))

    def inelastic(self):
        """  """
        return norm_swave * np.exp(1.j*phiins)*self.tin

    def dstn_ampl(self, mdd, md1pi):
        md2pi = self.mZsq(mdd, md1pi)
        return self.t * (self.bwdstp(md1pi) + self.bwdstp(md2pi))

    def calc(self, mdd, md1pi):
        return 2*mpip*self.KineC(mdd) * MagSq(self.a1(mdd, md1pi) + self.a2())

    def pdf_vars(self, e, pd, mdpi):
        """ Convenience function for PDF in terms of the observables
        Args:
          - e: energy (MeV)
          - pd: p(D) (MeV)
          - mdpi: m(Dpi) (MeV)
        """
        s, mddsq, md1pisq = vt.observables_to_mandelstam(e, pd, mdpi)
        return self.pdf_3d(s, mddsq, md1pisq)

    def pdf_3d(self, s, mddsq, md1pisq):
        """ 3D PDF
        Args:
          - s: Mandelstam variable (GeV^2)
          - mddsq: m^2(DD) (GeV^2)
          - md1pisq: m^2(Dpi) (GeV^2)
        """
        mask = self.isInPhsp(s, mddsq, md1pisq)
        result = np.zeros(mask.shape, dtype=float)
        if not np.any(mask):
            print('Warning: all zeros')
            result

        mMo = np.sqrt(s)
        E = mMo - TMtx.thr
        mMov, Ev, sv, mddsqv, md1pisqv = [x[mask] for x in [mMo, E, s, mddsq, md1pisq]]

        # pre-calculations
        md2pisqv = self.mijsq(sv, mddsqv, md1pisqv)
        # T-matrix factor
        tv = np.sum(self.tmtx(Ev)[0], axis=0)
        # Dalitz plot amplitude
        amp = tv * (self.bwdstp(md1pisqv) + self.bwdstp(md2pisqv))
        # pion momentum squared
        kine = 2 * mpip * self.iKineFramejk(mMov, mpip, mddsqv)
        # Full amplitude
        result[mask] = kine * MagSq(amp)
        return result

    def pdf(self, **kwargs):
        """ 3D or 5D PDF
            Required agruments:
                - mddsq
                - md1pisq
                - s or E
            Optional arguments:
                - gsre
                - gsim
        """
        assert 'mddsq' in kwargs and 'md1pisq' in kwargs
        assert 's' in kwargs or 'E' in kwargs

        mddsq, md1pisq = [kwargs[key] for key in ['mddsq', 'md1pisq']]
        if 's' in kwargs:
            s = kwargs['s']
            E = vt.s_to_e(s)
        else:
            E = kwargs['s']
            s = vt.e_to_s(E)

        mask = self.isInPhsp(s, mddsq, md1pisq)
        assert np.any(mask)

        result = np.zeros(mask.shape, dtype=float)
        Ev, sv, mddsqv, md1pisqv = [x[mask] for x in [E, s, mddsq, md1pisq]]

        mMov = np.sqrt(sv)
        if 'gsre' in kwargs or 'gsim' in kwargs:
            gsre = kwargs['gsre'][mask] if 'gsre' in kwargs else np.ones(sv.shape) * gs.real
            gsim = kwargs['gsim'][mask] if 'gsim' in kwargs else np.ones(sv.shape) * gs.imag
        
            gsv = gsre + 1j*gsim
            gtv = gt.real + 1j*gsim
            self.tmtx.set_gs_gt(gsv, gtv)

        # pre-calculations
        md2pisqv = self.mijsq(sv, mddsqv, md1pisqv)

        # T-matrix factor
        tv = np.sum(self.tmtx(Ev)[0], axis=0)

        # Dalitz plot amplitude
        amp = tv * (self.bwdstp(md1pisqv) + self.bwdstp(md2pisqv))

        # pion momentum squared
        kine = 2 * mpip * self.iKineFramejk(mMov, mpip, mddsqv)

        # Full amplitude
        result[mask] = kine * MagSq(amp)
        return result


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
