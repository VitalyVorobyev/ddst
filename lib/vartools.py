""" Variables transformation tools """

import numpy as np

from .params import mdn, mdstp

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
    """ (E, p(D), m(Dpi+)) (MeV) -> (s, m^2(DD), m^2(Dpi+)) (GeV) """
    return (
        (e + mdn + mdstp)**2,         # s
        (pd**2**3 / mdn + 2*mdn)**2,  # m^2(DD)
        md1pi**2                      # m^2(Dpi+)
    )
