#! /usr/bin/env python

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from mcfit import ConvFitterNorm
from lib.convolution import smear_1d

def smear(sample: np.ndarray, covar: callable, gen=None):
    if gen is None:
        gen = np.random.Generator(np.random.PCG64())

    return np.fromiter(
        map(lambda x: gen.normal(x, covar(x)), sample),
        dtype=np.float)


class NormPdf:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x:float):
        return stats.norm.pdf(x, self.mean, self.sigma)


def norm_1d_fit():
    """ Test of the fit procedure with 1D normal distribution """
    mean = 2.
    sigma = 0.5
    nsig = 2000
    nnorm = 20000

    pdf = NormPdf(mean, sigma)
    covar = lambda x: np.ones(x.size) * 0.7 + 0.1*x
    # covar = lambda x: np.ones(x.size) * 0.7

    gen = np.random.Generator(np.random.PCG64())
    sample = gen.normal(mean, sigma, nsig)
    smeared_sample = smear(sample, covar, gen)

    norm_sample = gen.uniform(smeared_sample.min(), smeared_sample.max(), nnorm)

    nbins=40
    plt.figure(figsize=(8,6))
    plt.hist(sample, bins=nbins, histtype='step', density=True)
    plt.hist(smeared_sample, bins=nbins, histtype='step', density=True)
    plt.hist(norm_sample, bins=nbins, histtype='step', density=True)
    plt.grid()
    plt.tight_layout()

    x = np.linspace(smeared_sample.min(), smeared_sample.max(), 100)
    pdfval = pdf(x)
    smpdfval = np.fromiter(map(lambda val: smear_1d(val, pdf, covar), x), dtype=np.float)
    print(pdfval.sum())
    print(smpdfval.sum())
    plt.plot(x, pdfval)
    plt.plot(x, smpdfval / smpdfval.sum() * pdfval.sum())
    plt.show()
    return

    fitter = ConvFitterNorm(pdf, covar)
    fmin, params, corrmtx = fitter.fit_to(smeared_sample, norm_sample, mean, sigma)
    print(fmin)
    print(corrmtx)


def main():
    norm_1d_fit()


if __name__ == "__main__":
    main()
