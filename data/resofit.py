""" """

import sys
sys.path.append('lib')
sys.path.append('data')

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from numpy.polynomial.polynomial import polyval

import matplotlib.pyplot as plt

from iminuit import Minuit

from params import datafile, mdn, mdstp
from plots import load_tree, make_hist

# Minuit helpers
wolim = lambda name, mean, err, fixed: {
            name: mean, 'error_'+name: err, 'fix_'+name: fixed}
wlim = lambda name, mean, err, lim, fixed:\
    dict({'limit_'+name: lim}, **wolim(name, mean, err, fixed))

def normed(f, x, r):
    """ """
    return f(x) / quad(f, r[0], r[1])[0]

def gauss_and_poly(x, mean, width, fgaus, r, p):
    """ """
    sig = lambda x: norm.pdf(x, mean, width)
    bkg = lambda x: polyval(x, p)
    return fgaus * normed(sig, x, r) + (1. - fgaus) * normed(bkg, x, r)

def dn_fit(x, r=(1.8, 1.925), mean=mdn, width=0.005, poly2=False):
    """ """
    print(f'{len(x)} events to fit')

    def fcn(f, m, w, p0, p1, p2):
        pdf = lambda x: gauss_and_poly(x, m, w, f, r, [p0, p1, p2])
        loglh = np.sum(-np.log(pdf(x)))
        print('loglh {:.3f}'.format(loglh))
        return loglh

    pd = dict({
        'errordef': 0.5,
        'forced_parameters': ['f', 'm', 'w', 'p0', 'p1', 'p2'],
        'fcn': fcn,
         **wlim('f', 0.8, 0.1, [0., 1.], False),
        **wolim('m', mean, 0.001, False),
         **wlim('w', width, 0.001, [0.00001, 0.010], False),
        **wolim('p0',  1.0, 0.1, False),
        **wolim('p1', -0.5, 0.1, False),
        **wolim('p2', 0., 0.1, not poly2)
    })

    minimizer = Minuit(**pd)
    fmin, param = minimizer.migrad()
    corrmtx = minimizer.matrix(correlation=False)
    return (fmin, param, corrmtx)

def main():
    if True:
        r=(1.8, 1.925)
        x = load_tree()['mass_C1c']
        mean, width = mdn, 0.005
    else:
        r=(2.006, 2.016)
        x = load_tree()['mass_C2c']
        mean, width = mdstp, 0.0005

    x = x[(x>r[0]) & (x<r[1])]
    fmin, param, corrmtx = dn_fit(x, r, mean, width)
    print(fmin)
    print(param)

    pdf = lambda x: gauss_and_poly(x,
        param[1].value,
        param[2].value,
        param[0].value,
        r,
        [param[3].value, param[4].value, param[5].value])
    xv = np.linspace(r[0], r[1], 150)

    dbins, dhist, errors = make_hist(x, range=r)
    norm = len(x) * (r[1] - r[0]) / len(dbins)

    print('Width: {:.2f} MeV'.format(param[2].value * 1000))

    plt.figure(figsize=(8,6))
    plt.errorbar(dbins, dhist, errors, linestyle='none')
    plt.plot(xv, norm*pdf(xv))
    plt.xlim(*r)
    plt.ylim(0, 1.05*max(dhist))
    plt.xlabel(r'$m(D^0), GeV$', fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.show()

def resolution_calculator(smdstp, smdn):
    """ """
    

if __name__ == '__main__':
    # main()
    resolution_calculator(0.36, 8.16)
