""" Variables transformation tools """

import numpy as np

from .params import mdn, mdstp, scale

def generated_to_observables(mddsq, md1pisq):
    """ (m^2(DD), m^2(D0pi+)) (GeV) -> (p(D), m(D0pi+)) (MeV)
        Cuts off negative kinetic energy (not a clean approach) """
    tdd = np.clip((np.sqrt(mddsq)*scale - 2*mdn), 0, None)
    return (np.sqrt(tdd * mdn), np.sqrt(md1pisq)*scale)

def e_to_s(e):
    return (e*10**-3 + mdn + mdstp)**2

def p_to_mddst(p):
    return ((p*10**-3)**2 / mdn + 2.*mdn)**2

def mdpi_to_mdpisq(mdpi):
    return (mdpi*10**-3)**2

coef = 10**-3*scale

def observables_to_mandelstam(e, pd, md1pi):
    """ (E, p(D), m(Dpi+)) (MeV) -> (s, m^2(DD), m^2(Dpi+)) (GeV) """
    return (
        (e*coef + mdn + mdstp)**2,         # s
        (pd**2*coef**3 / mdn + 2*mdn)**2,  # m^2(DD)
        (md1pi*coef)**2                    # m^2(Dpi+)
    )
