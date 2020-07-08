""" """

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from .params import sigma_ppi, sigma_mdn, mpip, mdn, mdstp

sppiSq = sigma_ppi**2
smdSq = sigma_mdn**2

def spline(x, y, newx):
    """ Cubic spline """
    return interp1d(x, y, kind='cubic')(newx).reshape(1, -1)

def smdstp(tpi=6.6e-3):
    """ sigma(D*+) """
    return jnp.sqrt(2*tpi/mpip)*sigma_ppi

def smddpi(e, tdd):
    """ sigma(m_DDpi) """
    tpi = e - mdn + mdstp - mpip
    return (2*mdn) / (2*mdn+mpip) *\
        jnp.sqrt(0.5*tdd/mdn * smdSq + 2*tpi/mpip*sppiSq)

def stdd(tdd):
    """ sigma(m_DD) """
    return jnp.sqrt(2*tdd/mdn)*sigma_mdn

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

def smear_e_const(e, ev, tdd=None, ptdd=None, dots=250, sigma=0.00035):
    """ """
    return (norm.pdf(ev, e, sigma), sigma)

@jax.jit
def sample(events: np.ndarray) -> (np.ndarray):
    """ Resolution sampler for MC events
    Args:
        - events: [E (MeV), m^2(DD) (GeV^2), m^2(Dpi) (GeV^2)]
    """
    e = events[:,0] * 10**-3
    tdd = jnp.sqrt(events[:,1]) - 2*mdn
    mdpi = jnp.sqrt(events[:,2])

    offsets = jax.random.normal(jax.random.PRNGKey(1), events.shape)

    return jnp.column_stack([
        (e + smddpi(e, tdd) * offsets[:,0]) * 10**3,
        (tdd + stdd(tdd) * offsets[:,1] + 2*mdn)**2,
        (mdpi + smdstp() * offsets[:,2])**2,
    ])
