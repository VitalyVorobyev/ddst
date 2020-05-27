""" """

import sympy as sp

gs, gt, d1, d2, mu1, mu2 = sp.symbols('gs gt d1 d2 mu1 mu2')

def k(d, mu, E):
    return sp.sqrt(sp.S(2)*mu*sp.sqrt(E-d))

def det(E):
    """ """
    k1 = k(d1, mu1, E)
    k2 = k(d2, mu2, E)
    return gs*gt - k1*k2 + sp.S(0.5j)*(gs+gt)*(k1+k2)

def fcn(E):
    d = det(E)
    return sp.re(d)**2 + sp.im(d)**2

def main():
    """ """
    E = sp.Symbol('E')
    solution = sp.solvers.solve(fcn(E), E)
    print(solution)

if __name__ == '__main__':
    main()
