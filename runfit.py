#! /usr/bin/env python

import os
import sys
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from mcfit import ConvFitterNorm
from lib import convolution as cnv
from lib.dndnpip import DnDnPip
from lib import params as prm
from lib import vartools
from lib.resolution import spd, smddpi2, smdstp
from lib import plots

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

def xcovar_3dvec(ev, pdv, md1piv):
    return np.stack([xcovar_3d(e, pd, md1pi) for e, pd, md1pi in
        zip(ev.ravel(), pdv.ravel(), md1piv.ravel())]).reshape(ev.shape + (3, 3))

# xcovar_3dvec = np.vectorize(xcovar_3d)

def norm_1d_fit():
    """ Test of the fit procedure with 1D normal distribution """
    mean = 2.
    sigma = 0.5
    nsig = 5000
    nnorm = 50000

    pdf = NormPdf(mean, sigma)
    coef = float(sys.argv[1]) if len(sys.argv) == 2 else 0.
    covar = lambda x: covariance(x, coef)

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

    if True:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(14.5, 9))
        for i, a in enumerate(ax[0]):
            a.hist(sample[:,i], bins=100, histtype='step', density=True)
            a.grid()
        for i, a in enumerate(ax[1]):
            a.scatter(sample[:,(i+0)%3], sample[:,(i-1)%3], s=0.4)
            a.grid()
        fig.tight_layout()

    data_box = cnv.build_box(sample)
    # print(data_box)
    bins = np.ones(data_box.shape[0], dtype=np.int32) * 256
    box_ticks = cnv.ticks_in_box(data_box, bins)
    box_grid = cnv.grid_in_box(data_box, bins)
    # print(list(map(lambda x: x.shape, box_grid)))

    # mand = vartools.observables_to_mandelstam(*box_grid)
    # print(list(map(lambda x: x*1e-6, mand)))

    gs = 40 + 1.5j
    gt = 25000 + 1.5j
    pdf = DnDnPip(gs, gt)
    f = pdf.pdf_vars(*box_grid)
    print(f.shape)

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(14.5, 4.5))
    plots.draw_pdf_projections(ax, box_ticks, f)
    fig.tight_layout()
    plt.show()
    

    # convolve_nd(
    #     data_box,
    #     lambda x: pdf.pdf_vars(x),
    #     xcovar_3dvec,
    #     np.ones(data_box.shape[0], dtype=np.int32) * nbins
    # )

    # item = 10
    # cov = xcovar_3d(sample[item,0], sample[item,1], sample[item,2])
    # print(np.linalg.inv(cov))

    # covs = xcovar_3dvec(sample[:,0], sample[:,1], sample[:,2])
    # print(covs.shape)
    # print(np.linalg.inv(covs)[item])

    # mgrid = meshgrid([sample[:,0], sample[:,1], sample[:,2]])
    # pdf_vals = pdf.pdf_vars(mgrid[:,0], mgrid[:,1], mgrid[:,2])
    # print(pdf_vals.shape)


def main():
    # norm_1d_fit()
    fit_model()


if __name__ == "__main__":
    main()
