""" """

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

import matplotlib.pyplot as plt

from params import *

smdstp = 0.36 * 10**-3
smd = 8.2 * 10**-3
smdstpSq = smdstp**2
smdSq = smd**2

def spline(x, y, newx):
    """ Cubic spline """
    return interp1d(x, y, kind='cubic')(newx).reshape(1, -1)

C1 = (2*mdn) / (2*mdn+mpip)
def smddpi(tdd):
    """ sigma(m_DDpi) """
    return C1 * np.sqrt(2*tdd/mdn * smdSq + smdstpSq)

def stdd(tdd):
    """ sigma(m_DD) """
    return np.sqrt(2*tdd/mdn)*smd

def smear_tdd(tdd, p, dots=250):
    """ """
    newx = np.linspace(tdd[0], tdd[-1], dots)
    xr, yr = np.meshgrid(newx, newx)
    r = norm.pdf(xr, yr, stdd(yr))
    r /= np.sum(r, axis=0)
    return (newx, np.sum(spline(tdd, p, newx) @ r, axis=0))

def smear_mdpi(mdpi, p, dots=250):
    """ """
    # newx = np.linspace(mdpi[0], mdpi[-1], dots)
    newx = np.linspace(mdpi[0], 2.020, dots)
    appx = newx[newx>mdpi[-1]]
    mdpi = np.append(mdpi, appx)
    p = np.append(p, np.zeros(appx.shape))
    xr, yr = np.meshgrid(newx, newx)
    r = norm.pdf(xr, yr, smdstp)
    r /= np.sum(r, axis=0)
    return (newx, np.sum(spline(mdpi, p, newx) @ r, axis=0))

def smear_e(e, ev, tdd, ptdd, dots=250):
    """ e: float - current energy
        ev: np.array - energy linspace
        tdd - T(DD)
        ptdd - T(DD) pdf
    """
    newtdd = np.linspace(tdd[0], tdd[-1], dots)
    newptdd = spline(tdd, ptdd, newtdd).flatten()
    er, tddr = np.meshgrid(ev, newtdd)
    r = norm.pdf(er, e, smddpi(tddr))
    r /= np.sum(r, axis=0)
    return (np.sum(newptdd.reshape(1,-1) @ r, axis=0),
            np.average(smddpi(newtdd.T), weights=newptdd))

def main():
    """ Unit test """
    dot1, dot2 = 250, 250
    tdd = np.linspace(0, 8, dot1)[1:]*10**-3
    p = norm.pdf(tdd, 0.00, 0.002)
    p /= np.sum(p) / dot1 * dot2
    x, y = smear_tdd(tdd, p, dot2)
    # plt.plot(tdd*10**3, stdd(tdd)*10**3)
    # plt.show()
    plt.plot(tdd, p)
    plt.plot(x, y)
    print(np.sum(p))
    print(np.sum(y))
    plt.show()

if __name__ == '__main__':
    tdd = np.linspace(0, 8, 25)[1:]*10**-3
    print(smddpi(tdd)*10**3)
    # for x in range(6):
    #     x = (x+1)*10**-3
    #     print(x*10**3, stdd(x)*10**3)
    #     lbl = 'T: {:.0f} MeV, S: {:.2f} MeV'.format(x*10**3, stdd(x)*10**3)
    #     plt.plot(tdd*10**3, norm.pdf(tdd, x, stdd(x)), label=lbl)
    # plt.xlim(0, 8)
    # plt.ylim(0, 1500)
    # plt.xlabel(r'$T_{DD}$, MeV', fontsize=16)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    # main()