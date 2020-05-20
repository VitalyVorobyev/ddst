""" The X(3873) -> D0 D0bar pi0 decay PDF """

import numpy as np

from dalitzphsp import DalitzPhsp
from params import *

class DDPi(DalitzPhsp):
    """ The X(3873) -> D0 D0bar pi0 decay PDF """
    checkEBalance = True

    def __init__(self, E):
        """ Constructor.
            E is the energy deposition in the X(3872) -> D0 anti-D*0 decay """
        super(DDPi, self).__init__(E + 2. * mD + dDstDpi, mD, mD, mpi)
        self.setE(E)

    def setE(self, E):
        self.E = E
        self.setM(E + 2. * mD + mpi + dDstDpi)
        self.phspCoef = 1. / self.moSq
        
        # precalculation for propagator
        self.bwden = mu * (2.*E + 1.j*Gamma)
        print('##### DDPi: E {:.3f} #####'.format(E / unit))
        print('  mX:  {:.3f}'.format(self.mo))

    def __calc(self, mdd, mdpi, mdbpi):
        """ Decay probability density calculation """
        mask = self.inPhspABBC(mdd, mdbpi)
        # print(np.count_nonzero(mask))

        td    = self.KineA(mdbpi) if isinstance(mdbpi, float) else self.KineA(mdbpi[mask])
        tdbar = self.KineB(mdpi)  if isinstance(mdpi,  float) else self.KineB(mdpi[mask])
        tpi   = self.KineC(mdd)   if isinstance(mdd,   float) else self.KineC(mdd[mask])

        if DDPi.checkEBalance:
            assert(np.allclose(td + tdbar + tpi, self.E + dDstDpi))

        result = np.zeros(len(mask), dtype=float)
        mc = 1. / (2.*mD * td    - self.bwden) +\
             1. / (2.*mD * tdbar - self.bwden)
        result[mask] = tpi * self.phspCoef * (mc.real**2 + mc.imag**2)
        return (result, mask)

    def __call__(self, **kwargs):
        """ Decay probability density. A pair of Dalitz variables must be provided:
               (m^2(DDbar), m^2(D pi0)),
               (m^2(DDbar), m^2(Dbar pi0)) or
               (m^2(D pi0), m^2(Dbar pi0))
            The argument names are: 'mdd', 'mdpi' and 'mdbpi'
            Returns tuple of
                - PDF values (np.array of floats)
                - phase space mask (np.array of bool)
        """
        if 'mdd' in kwargs and 'mdpi' in kwargs:
            mdd, mdpi, mdbpi = kwargs['mdd'], kwargs['mdpi'], self.mZsq(kwargs['mdd'], kwargs['mdpi'])
        elif 'mdpi' in kwargs and 'mdbpi' in kwargs:
            mdd, mdpi, mdbpi = self.mZsq(kwargs['mdpi'], kwargs['mdbpi']), kwargs['mdpi'], kwargs['mdbpi']
        elif 'mdd' in kwargs and 'mdbpi' in kwargs:
            mdd, mdpi, mdbpi = kwargs['mdd'], self.mZsq(kwargs['mdd'], kwargs['mdbpi']), kwargs['mdbpi']
        else:
            print('DDGam: wrong agrumnets {}'.format(kwargs))
            assert(False)
        return self.__calc(mdd, mdpi, mdbpi)
