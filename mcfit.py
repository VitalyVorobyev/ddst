""" Max likelihood fit of the smeared MC sample """

import os
import numpy as np

from iminuit import Minuit

from lib.convolution import smear_nd, smear_1d
from lib.vartools import generated_to_observables, observables_to_mandelstam
from lib.dndnpip import DnDnPip

import lib.params as pars

path_data = './mcsamples'

def sample_fname(re, im ,ch):
    """ Smeared toy MC data set file name """
    fname = os.path.join(
        path_data, f'mc_ddpip_3d_gs{re:.2f}_{im:.2f}_ch{ch}_smeared.npy')
    if os.path.isfile(fname):
        return fname
    print(f'file {fname} not found')
    return None

def get_sample(re, im, ch, nevt):
    """ Load smeared toy MC data set """
    fname = sample_fname(re, im, ch)
    if fname:
        data = np.load(fname)[:nevt]
        return (data[:, 0], *generated_to_observables(data[:, 1], data[:, 2]))
    else:
        return None


class ConvFitterMC():
    def __init__(self, pdf):
        self.pdf = pdf

    def fcn(self, gsre, gsim, gtre):
        self.pdf.set_gs_gt(
            (gsre + 1j*gsim)*10**-3,
            (gtre + 1j*gsim)*10**-3
        )
        loglh = self.loglh()
        print(f'loglh {loglh:.3f}')
        return loglh

    def loglh(self):
        return -np.sum(np.log(self.pdf(self.data))) +\
            np.log(np.sum(self.pdf(self.norm_sample))) * self.data.shape[0]

    def fit_to(self, data, norm_sample, gsre0, gsim0, gtre0):
        self.data = data
        self.norm_sample = norm_sample

        mnt = Minuit(self.fcn, errordef=0.5,
            gsre=gsre0, error_gsre=1., limit_gsre=(-10, 60), fix_gsre=False,
            gsim=gsim0, error_gsim=1., limit_gsim=(0.5, 2.0), fix_gsim=True,
            gtre=gtre0, error_gtre=1., limit_gtre=(-10, 60), fix_gtre=True,
        )

        fmin, params = mnt.migrad()
        mnt.print_param()
        corrmtx = mnt.matrix(correlation=True)
        return (fmin, params, corrmtx)


class ConvFitterNorm():
    def __init__(self, pdf: callable, covar: callable):
        """
        Args:
            - pdf: true, not smeared, PDF
            - covar: returns covariance matrix for a given point
        """
        self.pdf = pdf
        self.covar = covar

    def fcn(self, mean, sigma):
        self.pdf.mean = mean
        self.pdf.sigma = sigma
        loglh = self.loglh()
        print(f'loglh: {loglh:.3f} mean: {mean:.3f} sigma: {sigma:.3f}')
        return loglh

    def loglh(self):
        smpdf = np.fromiter(
            map(lambda x: smear_1d(x, self.pdf, self.covar), self.data),
            dtype=np.float
        )
        return -np.sum(np.log(smpdf)) +\
            np.log(np.sum(self.pdf(self.norm_sample))) * self.data.shape[0]
        

    def fit_to(self, data, norm_sample, mean0, sigma0):
        self.data = data
        self.norm_sample = norm_sample

        mnt = Minuit(self.fcn, errordef=0.5,
            mean=mean0, error_mean=1., limit_mean=(-5., 5.), fix_mean=False,
            sigma=sigma0, error_sigma=1., limit_sigma=(0.1, 2.0), fix_sigma=False
        )

        fmin, params = mnt.migrad()
        mnt.print_param()
        corrmtx = mnt.matrix(correlation=True)
        return (fmin, params, corrmtx)
