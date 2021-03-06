""" [Hanhart, arXiv:1602.00940] """

import numpy as np
from .params import gs, gt, mdn, mpin, mdstn, mdstnSq, mdp, mdstp
from .params import gamma_star_n_dngam, gamma_star_n_dnpin, delta000, gamma_star_n

def rmass(m1, m2):
    """ Reduced mass """
    return m2*m1/(m2+m1)

def RelativisticBreitWigner(s, m, G):
    """ Relativistic Breit-Wigner """
    return 1. / (m**2 - s -1j*m*G)

def DstnWidth(s):
    """ D*0 with energy-dependent width """
    deltaE = np.sqrt(s) - mdn - mpin
    deltaE[deltaE<0] = 0
    return gamma_star_n_dngam + gamma_star_n_dnpin *\
        (deltaE / delta000)**(1.5)

def RbwDstn(s):
    """ D*0 RWB with energy-dependent width """
    return 1. / (mdstnSq - s - 1j*mdstn*DstnWidth(s))
    # return 1. / (mdstnSq - s - 1j*mdstn*gamma_star_n)

def MagSq(z):
    return z.real**2 + z.imag**2

class TMtx():
    """ T-matrix for D*0D+ D*+D0 channels """
    mu = rmass(mdn, mdstp)
    thr = mdn + mdstp
    d = [mdn + mdstp - thr, mdp + mdstn - thr]

    def __init__(self, gs, gt):
        """ gs, gt can be float or np.array """
        self.set_gs_gt(gs, gt)

    def set_gs_gt(self, gs, gt):
        """ """
        self.gs, self.gt = gs, gt
        self.gsumm = gt + gs
        self.gdiff = gt - gs
        self.gprod = gt * gs

    def k(self, E, idx):
        """ Two-body momentum of the DD* system """
        return np.sqrt((2.+0j) * TMtx.mu * (E-TMtx.d[idx]))

    def det(self, k1, k2):
        """ T-matrix determinant  """
        return self.gprod - k1*k2 + 0.5*1j*self.gsumm*(k1+k2)

    def t11(self, k):
        """ T[1,1] or T[2,2] (1-based) """
        return 0.5 * self.gsumm + 1j*k

    def t12(self):
        """ T[1,2] = T[2,1] (1-based) """
        return 0.5 * self.gdiff

    def __call__(self, E):
        """ """
        k1, k2 = [self.k(E, idx) for idx in [0,1]]

        offdiag = self.t12()
        if not isinstance(E, (int, float)):
            offdiag = offdiag * np.ones(len(E))

        return np.array([
            [self.t11(k2), offdiag],
            [offdiag, self.t11(k1)]
        ]) / self.det(k1,k2)
