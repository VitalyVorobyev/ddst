""" Variables transformation tools """

import numpy as np

from .params import mdn, mdstp

def generated_to_observables(mddsq, md1pisq):
    """ (m^2(DD), m^2(D0pi+)) (GeV) -> (p(D), m(D0pi+)) (MeV)
        Cuts off negative kinetic energy (not a clean approach) """
    tdd = np.clip((np.sqrt(mddsq) - 2*mdn)*10**3, 0, None)
    return (np.sqrt(tdd * mdn * 10**3), np.sqrt(md1pisq) * 10**3)

def e_to_s(e):
    return (e*10**-3 + mdn + mdstp)**2

def p_to_mddst(p):
    return ((p*10**-3)**2 / mdn + 2.*mdn)**2

def mdpi_to_mdpisq(mdpi):
    return (mdpi*10**-3)**2

def observables_to_mandelstam(e, pd, md1pi):
    """ (E, p(D), m(Dpi+)) (MeV) -> (s, m^2(DD), m^2(Dpi+)) (GeV) """
    return (
        (e*10**-3 + mdn + mdstp)**2,      # s
        (pd**2*10**-6 / mdn + 2*mdn)**2,  # m^2(DD)
        (md1pi*10**-3)**2                 # m^2(Dpi+)
    )
