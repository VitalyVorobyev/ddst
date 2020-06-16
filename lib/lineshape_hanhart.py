""" [Hanhart, arXiv:1602.00940] """

import numpy as np
from .params import gs, gt, mdn, mpin, gamma_star_n_dngam, gamma_star_n_dnpin, mdstn, mdstnSq, delta000, mdp, mdstp

def rmass(m1, m2):
    """ Reduced mass """
    return m2*m1/(m2+m1)

def RelativisticBreitWigner(s, m, G):
    """ Relativistic Breit-Wigner """
    return 1. / (m**2 - s -1j*m*G)

def DstnWidth(s):
    """ D*0 with energy-dependent width """
    return gamma_star_n_dngam + gamma_star_n_dnpin *\
        ((np.sqrt(s) - mdn - mpin) / delta000)**(1.5)

def RbwDstn(s):
    """ D*0 RWB with energy-dependent width """
    return 1. / (mdstnSq - s - 1j*mdstn*DstnWidth(s))

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

def main():
    """ Unit test """
    print('Lower threshold {:.3f} MeV'.format(TMtx.thr*10**3))
    print(' delta Threshold {:.3f} MeV'.format((TMtx.d[1] - TMtx.d[0])*10**3))

    t = TMtx(gs, gt)
    E = -0.225*10**-3
    k1 = t.k(E, 0)
    k2 = t.k(E, 1)
    print(f'k1 {k1*10**3}')
    print(f'k2 {k2*10**3}')
    print(f'det {t.det(k1, k2)*10**6}')
    print(f't11 {t.t11(k2)*10**3}')
    print(f't22 {t.t11(k1)*10**3}')
    print(f't12 {t.t12()*10**3}')
    print(f'2mu {t.mu*2}')
    print(MagSq((t.t11(k2) + t.t12()) / t.det(k1, k2)) * 10**-6)

    import matplotlib.pyplot as plt
    plt.figure()
    E = np.linspace(mdn + mpin, mdstn+2e-3)
    plt.plot((E - mdstn)*10**3, DstnWidth(E**2)*10**6)
    plt.xlabel('E (MeV)', fontsize=14)
    plt.ylabel('D*0 width (keV)', fontsize=14)
    plt.ylim(0, 80)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
