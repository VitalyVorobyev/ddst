""" [Hanhart, arXiv:1602.00940] """

import numpy as onp
import numpy as np
# import jax.numpy as np
from params import *

def rmass(m1, m2):
    """ Reduced mass """
    return m2*m1/(m2+m1)

def RelativisticBreitWigner(s, m, G):
    """ Relativistic Breit-Wigner """
    return 1. / (m**2 - s -1j*m*G)

def MagSq(z):
    return z.real**2 + z.imag**2

class TMtx(object):
    """ T-matrix for D*0D+ D*+D0 channels """
    mu = rmass(mdn, mdstp)
    mu2 = rmass(mdp, mdstn)
    thr = mdn + mdstp
    d = [mdn + mdstp - thr, mdp + mdstn - thr]

    def __init__(self, gs, gt):
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
        return np.array([
            [self.t11(k2), self.t12()],
            [self.t12(),   self.t11(k1)]
        ]) / self.det(k1,k2)

    def summary(self):
        print('t pole {0.real:.3f} + {0.imag:.3f}'.format(self.pole(self.gt, gamma_star_p)))
        print('s pole {0.real:.3f} + {0.imag:.3f}'.format(self.pole(self.gs, gamma_star_z)))
        print('Thr lo {:.3f}'.format(TMtx.d[0]))
        print('Thr hi {:.3f}'.format(TMtx.d[1]))

    def pole(self, g, w):
        """ Complex energy: -Ex -0.5j*Gammax. The pole position """
        return -(self.Ex(g) + 0.5j*self.Gx(g,w)) / unit

    def Ex(self, g):
        """ Real part of pole position - the binding energy """
        return 0.5 * (g.real**2 - g.imag**2) / TMtx.mu

    def Gx(self, g, width):
        """ Imag part of pole position - the binding width """
        return width + 2.*g.real*g.imag / TMtx.mu

def main():
    """ Unit test """
    print('Lowest threshold {:.3f} MeV'.format(TMtx.thr*10**3))
    print(' delta Threshold {:.3f} MeV'.format((TMtx.d[1] - TMtx.d[0])*10**3))

    gs = (30 + 1.j) * 10**-3
    gt = (30 + 1.j) * 10**-3
    t = TMtx(gs, gt)

    E = (-2. + 10.*np.random.rand(100))*10**-3
    dE = TMtx.d[1] - TMtx.d[0]
    assert np.allclose(t.t12(), np.zeros(E.shape))
    assert np.allclose(t.t11(t.k(E, 0)), t.t11(t.k(E+dE, 1)))

if __name__ == '__main__':
    main()
