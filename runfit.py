#! /usr/bin/env python

import sys
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from mcfit import ConvFitterNorm
from lib.convolution import smear_1d, convolve_1d

def smear(sample: np.ndarray, covar: callable, gen=None):
    if gen is None:
        gen = np.random.Generator(np.random.PCG64())

    return np.array([gen.normal(x, covar(x)) for x in sample])
    # return np.apply_along_axis(
    #     lambda x: gen.normal(x, covar(x)), axis=0, arr=sample)

class NormPdf:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x:float):
        return stats.norm.pdf(x, self.mean, self.sigma)

def covariance(x, coef):
    if isinstance(x, float):
        min(max(0.1, 0.7 + coef*x**2), 10)
    return np.clip(np.ones(x.size)*0.7 + coef*x**2, 0.1, 10)

def norm_1d_fit():
    """ Test of the fit procedure with 1D normal distribution """
    mean = 2.
    sigma = 0.5
    nsig = 5000
    nnorm = 50000

    pdf = NormPdf(mean, sigma)
    coef = float(sys.argv[1]) if len(sys.argv) == 2 else 0.
    covar = lambda x: covariance(x, coef)
    # covar = lambda x: np.clip(np.ones(x.size) * 0.7 + coef*x, 0.1, 10)
    # covar = lambda x: max(0.1, 0.7*x + coef*x)

    gen = np.random.Generator(np.random.PCG64())
    sample = gen.normal(mean, sigma, nsig)
    smeared_sample = smear(sample, covar, gen)

    lo, hi = smeared_sample.min(), smeared_sample.max()
    norm_sample = gen.uniform(lo, hi, nnorm)

    nbins=40
    plt.figure(figsize=(8,6))
    plt.hist(sample, bins=nbins, histtype='step', density=True)
    plt.hist(smeared_sample, bins=nbins, histtype='step', density=True)
    plt.hist(norm_sample, bins=nbins, histtype='step', density=True)
    plt.grid()
    plt.tight_layout()

    x = np.linspace(lo, hi, 200)
    dx = x[1] - x[0]
    pdfval = pdf(x)
    smpdfval = smear_1d(x, pdf, covar)
    print(pdfval.sum() * dx)
    print(smpdfval.sum() * dx)
    xx, convpdf = convolve_1d(lo, hi, pdf, covar)
    plt.plot(x, pdfval)
    plt.plot(x, smpdfval)
    plt.plot(xx, convpdf)
    plt.show()

    fitter = ConvFitterNorm(pdf, covar)
    fmin, params, corrmtx = fitter.fit_to(smeared_sample, norm_sample, mean, sigma)
    print(fmin)
    print(corrmtx)


def main():
    norm_1d_fit()


if __name__ == "__main__":
    main()
