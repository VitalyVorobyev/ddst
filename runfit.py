#! /usr/bin/env python

import os
import sys
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from mcfit import ConvFitterNorm
from lib.convolution import smear_1d, convolve_1d, meshgrid, local_grid_nd
from lib.dndnpip import DnDnPip
from lib import params as prm
from lib import vartools
from lib.resolution import spd, smddpi2, smdstp

def smear(sample: np.ndarray, covar: callable, gen=None):
    if gen is None:
        gen = np.random.Generator(np.random.PCG64())
    return np.array([gen.normal(x, covar(x)) for x in sample])

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

def xcovar_3d(e, pd, md1pi):
    return np.diag([spd(), smddpi2(e, pd), smdstp()])**2

xcovar_3dvec = np.vectorize(xcovar_3d)


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

    print(f'{sample.mean():.3f} {sample.std():.3f}')
    print(f'{smeared_sample.mean():.3f} {smeared_sample.std():.3f}')

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


def fit_model():
    sample_path = 'mcsamples'
    sample_file = 'mc_ddpip_3d_gs40.00_1.50_ch10.npy'
    # sample_file = 'mc_ddpip_3d_gs40.00_1.50_ch10_smeared.npy'
    sample_raw = np.load(os.path.join(sample_path, sample_file))
    print(sample_raw.shape)

    pd, md1pi = vartools.generated_to_observables(
        sample_raw[:,1], sample_raw[:,2])

    sample = np.column_stack((sample_raw[:,0], pd, md1pi))
    eps = 1e-3
    sample = sample[(sample[:,1] > eps) & (sample[:,0] < 9) & (sample[:,1] < 120)]
    print(sample.shape)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(14.5, 9))
    for i, a in enumerate(ax[0]):
        a.hist(sample[:,i], bins=100, histtype='step')
        a.grid()
    for i, a in enumerate(ax[1]):
        a.scatter(sample[:,(i+0)%3], sample[:,(i-1)%3], s=0.4)
        a.grid()
    fig.tight_layout()
    # plt.show()

    print(sample[:,0].shape)
    sample_gev = sample * 1.e-3
    item = 10
    cov = xcovar_3d(sample_gev[item,0], sample_gev[item,1], sample_gev[item,2])
    print(np.linalg.inv(cov))

    # covs = xcovar_3dvec(sample[:,0], sample[:,1], sample[:,2])
    # print(covs.shape)

    # pdf = DnDnPip(prm.gs, prm.gt)
    # mgrid = meshgrid([sample[:,0], sample[:,1], sample[:,2]])
    # pdf_vals = pdf.pdf_vars(mgrid[:,0], mgrid[:,1], mgrid[:,2])
    # print(pdf_vals.shape)


def main():
    # norm_1d_fit()
    fit_model()


if __name__ == "__main__":
    main()