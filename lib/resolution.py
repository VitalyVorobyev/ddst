""" """

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
# import jax
# import jax.numpy as jnp

import matplotlib.pyplot as plt

from .params import sigma_ppi, sigma_mdn, mpip, mdn, mdstp, scale, tpi

sppiSq = sigma_ppi**2
smdSq = sigma_mdn**2

def spline(x, y, newx):
    """ Cubic spline """
    return interp1d(x, y, kind='cubic')(newx).reshape(1, -1)

def smdstp(tpi=tpi):
    """ sigma(D*+) """
    return np.sqrt(2*tpi/mpip)*sigma_ppi

def kine_pi(e):
    """ Kinetic energy of pi+ T(pi+) """
    return e - mdn + mdstp - mpip

def smddpi(e, tdd):
    """ sigma(m_DDpi) """
    return (2*mdn) / (2*mdn+mpip) *\
        np.sqrt(0.5*tdd/mdn * smdSq + 2*kine_pi(e)/mpip*sppiSq)

def smddpi2(e, pd):
    """ sigma(m_DDpi) in terms of p_D """
    return smddpi(e, pd**2 / mdn)
    # return (2*mdn) / (2*mdn+mpip) *\
        # jnp.sqrt(0.5*pd**2/mdn**2 * smdSq + 2*kine_pi(e)/mpip*sppiSq)

def stdd(tdd):
    """ sigma(m_DD) """
    return np.sqrt(2*tdd/mdn)*sigma_mdn

def spd():
    """ sigma(p_D) """
    return sigma_mdn / np.sqrt(2)

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
    newx = np.linspace(mdpi[0], 2.020*scale, dots)
    appx = newx[newx>mdpi[-1]]
    mdpi = np.append(mdpi, appx)
    p = np.append(p, np.zeros(appx.shape))
    xr, yr = np.meshgrid(newx, newx)
    r = norm.pdf(xr, yr, smdstp())
    r /= np.sum(r, axis=0)
    return (newx, np.sum(spline(mdpi, p, newx) @ r, axis=0))

def smear_e_fixed_tdd(ev, epdf, tdd):
    """ ev: np.array - energy linspace
        epdf: np.array - energy pdf
        tdd - T(DD)
    """
    result = np.zeros(ev.shape)
    for e, p in zip(ev, epdf):
        result += p * norm.pdf(ev, e, smddpi(e, tdd))
    return result

def smear_e(e, ev, tdd, ptdd, dots=250):
    """ e: float - current energy
        ev: np.array - energy linspace
        tdd - T(DD)
        ptdd - T(DD) pdf
    """
    newtdd = np.linspace(tdd[0], tdd[-1], dots)
    newptdd = spline(tdd, ptdd, newtdd).flatten()
    er, tddr = np.meshgrid(ev, newtdd)
    r = norm.pdf(er, e, smddpi(e, tddr))
    # r /= np.sum(r, axis=0)
    return (np.sum(newptdd.reshape(1,-1) @ r, axis=0),
            np.average(smddpi(e, newtdd.T), weights=newptdd))

def smear_e_const(e, ev, tdd=None, ptdd=None, dots=250, sigma=0.00035*scale):
    """ """
    return (norm.pdf(ev, e, sigma), sigma)

# @jax.jit
# def sample(events: np.ndarray) -> (np.ndarray):
#     """ Resolution sampler for MC events
#     Args:
#         - events: [E (MeV), m^2(DD) (GeV^2), m^2(Dpi) (GeV^2)]
#     """
#     e = events[:,0] * 10**-3
#     tdd = jnp.sqrt(events[:,1]) - 2*mdn
#     mdpi = jnp.sqrt(events[:,2])

#     offsets = jax.random.normal(jax.random.PRNGKey(1), events.shape)

#     return jnp.column_stack([
#         (e + smddpi(e, tdd) * offsets[:,0]) * 10**3,
#         (tdd + stdd(tdd) * offsets[:,1] + 2*mdn)**2,
#         (mdpi + smdstp() * offsets[:,2])**2,
#     ])

def sample(events: np.ndarray, seed=None) -> (np.ndarray):
    """ Resolution sampler for MC events
    Args:
        - events: [E (MeV), m^2(DD) (GeV^2), m^2(Dpi) (GeV^2)]
    """
    e = events[:,0] * 10**-3
    tdd = np.sqrt(events[:,1]) - 2*mdn
    mdpi = np.sqrt(events[:,2])

    offsets = np.random.default_rng(seed=seed).normal(loc=0, scale=1, shape=events.shape)

    return np.column_stack([
        (e + smddpi(e, tdd) * offsets[:,0]) * 10**3,
        (tdd + stdd(tdd) * offsets[:,1] + 2*mdn)**2,
        (mdpi + smdstp() * offsets[:,2])**2,
    ])

def get_resolution(e, pd):
    """ """
    return (smddpi2(e / 10**3, pd / 10**3)*10**3, spd()*10**3, smdstp()*10**3)


def merge(x1, y1, x2, y2, bins=5000):
    newx = np.linspace(min(x1[0], x2[0]), max(x1[-1], x2[-1]), bins)

    applo1 = newx[newx<x1[0]]
    x1 = np.append(applo1, x1)
    y1 = np.append(np.zeros(applo1.shape), y1)

    applo2 = newx[newx<x2[0]]
    x2 = np.append(applo2, x2)
    y2 = np.append(np.zeros(applo2.shape), y2)

    apphi1 = newx[newx>x1[-1]]
    x1 = np.append(x1, apphi1)
    y1 = np.append(y1, np.zeros(apphi1.shape))

    apphi2 = newx[newx>x2[-1]]
    x2 = np.append(x2, apphi2)
    y2 = np.append(y2, np.zeros(apphi2.shape))

    y1new = interp1d(x1, y1, kind='cubic')(newx)
    y2new = interp1d(x2, y2, kind='cubic')(newx)
    return (newx, y1new+y2new)
