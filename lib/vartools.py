""" Variables transformation tools """

import numpy as np

from .params import mdn, mdstp, mpip
from .resolution import spline

mu_dnpip = mdn * mpip / (mdn + mpip)  # reduced mass

def generated_to_observables(mddsq, md1pisq):
    """ (m^2(DD), m^2(D0pi+)) -> (p(D), m(D0pi+))
        Cuts off negative kinetic energy (not a clean approach) """
    tdd = np.clip((np.sqrt(mddsq) - 2*mdn), 0, None)
    return (np.sqrt(tdd * mdn), np.sqrt(md1pisq))

def e_to_s(e):
    return (e + mdn + mdstp)**2

def s_to_e(s):
    return np.sqrt(s) - mdn - mdstp

def p_to_mddst(p):
    return (p**2 / mdn + 2.*mdn)**2

def observables_to_mandelstam(e, pd, md1pi):
    """ (E, p(D), m(Dpi+)) -> (s, m^2(DD), m^2(Dpi+)) """
    return (
        (e + mdn + mdstp)**2,         # s
        (pd**2**3 / mdn + 2*mdn)**2,  # m^2(DD)
        md1pi**2                      # m^2(Dpi+)
    )

def mdpi_to_ppi(mdpi):
    """ m(D pi) -> p(pi) """
    return 2 * mu_dnpip * (mdpi - mdn - mpip)

def mdpisq_to_ppi(mdpisq):
    """ m^2(D pi) -> p(pi) """
    return mdpi_to_ppi(np.sqrt(mdpisq))

def msq_to_m(msq, wojac=False):
    """ m^2 -> m """
    if wojac:
        return np.sqrt(msq)
    m = np.sqrt(msq)
    return (m, jac_msq_to_m(m))

def jac_msq_to_m(m):
    """ Jacobian for m^2 -> m """
    return 2 * m

mddsq_min = 4*mdn**2
def mddsq_to_pd(mddsq, wojac=False):
    """ m^2(D0 D0) -> p(D0) """
    pd = np.sqrt(np.clip(mddsq - mddsq_min, 1e-6, a_max=None))
    return pd if wojac else (pd, jac_mddsq_to_pd(pd))

def jac_mddsq_to_pd(pd):
    """ Jacobian for m^2(D0 D0) -> p(D0) """
    return 0.5 / pd

def transform_distribution(x, pdf, transform, nbins=256):
    """ Transforms binned distribution taking jacobian into account """
    y, yjac = transform(x)
    yspace = np.linspace(y[0], y[-1], nbins)
    return (yspace, spline(y, pdf * yjac, yspace))

def tpi(mddpisq, mdd):
    """ Kinetic energy of pi+ in the D0D0 frame from m^2(DDpi) and m(DD) """
    return np.sqrt(mddpisq - mpip * (2*mdd - mpip)) - mdd
