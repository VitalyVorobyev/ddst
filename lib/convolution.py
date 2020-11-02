""" Tools for local numerical convolution with resolution """

import itertools
import numpy as np
from scipy import stats, signal

from typing import Iterable, Tuple, Callable

from . import resolution as res
from .params import mdn, mpip, mdstp


def get_resolution(e, pd):
    """  """
    return (
        res.smddpi2(e / 10**3, pd / 10**3)*10**3,
        res.spd()*10**3, 
        res.smdstp()*10**3
    )

def get_covariance(observables:np.ndarray) -> (np.ndarray):
    """ observables: shape (N, 3)
    Returns: np.array, shape(N, 3, 3)
    """
    return np.array([])

def local_grid_1d(x: float, sigma: float,
                  ndots: int=101, nsigma: float=5.) -> (Tuple):
    data_grid = np.linspace(x - nsigma*sigma, x + nsigma*sigma, ndots).flatten()
    delta = data_grid[1] - data_grid[0]
    reso_grid = np.arange(-nsigma*sigma, nsigma*sigma + 0.5*delta, delta)
    return (data_grid, reso_grid, delta)

def meshgrid(lists):
    # return np.fromiter(itertools.product(*lists), dtype=np.float)
    return np.array(list(itertools.product(*lists)))


def local_grid_nd(data: Iterable, sigma: Iterable,
                  ndots: int=101, nsigma: float=5.) -> (Tuple):
    grids = [local_grid_1d(x, s, ndots, nsigma) for x, s in zip(data, sigma)]
    return (meshgrid([item[0] for item in grids]),
            meshgrid([item[1] for item in grids]),
            np.prod([item[2] for item in grids]))

def norm(x, s):
    return np.exp(-0.5 * (x / s)**2) / ((2.*np.pi)**(0.5) * s)


def smear_1d(x:Iterable, pdf:Callable, sigma:Callable,
             ndots:int=101, nsigma:float=5) -> (float):
    """ Calculates smeared pdf value """
    result = []
    for item, sig in zip(x, sigma(x)):
        data_grid, reso_grid, delta = local_grid_1d(item, sig)
        result.append(np.dot(pdf(data_grid), norm(reso_grid[::-1], sigma(data_grid))) * delta)
    return np.array(result)


def convolve_1d(lo: float, hi:float, pdf:Callable, sigma:Callable, ndots=512):
    """ Global convolution """
    delta = hi - lo
    xgrid = np.linspace(lo, hi, ndots)
    ygrid = np.linspace(-delta, delta, 2*ndots)
    result = np.zeros(3*ndots)
    for idx, (p, s) in enumerate(zip(pdf(xgrid), sigma(xgrid))):
        result[idx:idx+2*ndots] += p*norm(ygrid, s)
    return (xgrid, result[ndots:-ndots] * delta / ndots)
    
def build_box(data:np.ndarray):
    """ Rectangular box containing all events. Arg. data: [N x ndim] """
    return np.array([[data[:,i].min(), data[:,i].max()] for i in range(data.shape[1])])

def ticks_in_box(box, binning):
    return [np.linspace(lo, hi, n) for [lo, hi], n in zip(box, binning)]

def grid_in_box(box, binning):
    """ """
    return np.meshgrid(*ticks_in_box(box, binning))

def build_reso_box(box):
    """ """
    return np.array([[lo - hi, hi - lo] for lo, hi in box])

def build_conv_box(box):
    """ """
    return np.array([[lo - (hi - lo), hi + (hi - lo)] for lo, hi in box])

def smear_1d_v0(x:float, pdf:Callable, sigma:Callable,
             rpdf:Callable=stats.norm.pdf, ndots:int=101,
             nsigma:float=5) -> (float):
    """ Calculates smeared pdf value """
    data_grid, reso_grid, delta = local_grid_1d(x, sigma(x))
    smeared_pdf = signal.fftconvolve(
        pdf(data_grid), rpdf(reso_grid, 0, sigma(reso_grid)), 'same') * delta
    return smeared_pdf[ndots // 2]

def vectorized_mvn(x:np.ndarray, cov:np.ndarray, mean:np.ndarray=None)\
        -> (np.ndarray):
    """ Calculates multivariate normal distribution with event-dependent
        covariance and mean
    TODO: implement event-dependent mean
    Args:
        - x shape (N, k)
        - cov shape (N, k, k)
        - mean shape (N, k)
    Returns:
        - (N, 1)
    """
    denominator = (2.*np.pi)**(x.shape[1] / 2) * np.linalg.det(cov)**0.5
    numerator = np.exp(
        -0.5 * np.einsum('...j, ...jk, ...k -> ...', x, np.linalg.inv(cov), x))
    return numerator / denominator


def convolve_nd(box:Iterable, pdf:Callable, covar:Callable, binning:Iterable):
    """ """
    main_grid = grid_in_box(box, binning)
    print(main_grid[0].shape, len(main_grid))
    mgrid = np.stack(main_grid, axis=-1)
    print(mgrid.shape)
    return
    covars = covar(*main_grid)

    conv_grid = grid_in_box(build_conv_box(box), 3*binning)
    reso_grid = grid_in_box(build_reso_box(box), 2*binning)


def smear_nd(x:Iterable, pdf:Callable, covar:Callable,
             ndots:int=31, nsigma:float=5) -> (float):
    """  """
    sigma = np.sqrt(np.diag(covar(np.array([x]))[0]))
    data_grid, reso_grid, delta = local_grid_nd(x, sigma)
    fres = vectorized_mvn(reso_grid, covar(reso_grid))
    return np.sum(pdf(data_grid) * np.flip(fres)) * delta

def local_resolution_grid(e, pd, mdpi, ndots=101, nsigma=5):
    """ Local grid and 3D Gaussian resolution window """
    sigmas = get_resolution(e, pd)
    x_full = [np.linspace(x - sig*nsigma, x + sig*nsigma, ndots)\
              for x, sig in zip([e, pd, mdpi], sigmas)]
    x_full_grid = np.array(list(itertools.product(*x_full)))
    dx = [xi[1] - xi[0] for xi in x_full]
    dr = dx[0] * dx[1] * dx[2]
    x_reso = [np.arange(-sig*nsigma, sig*nsigma + 0.5*step, step)
        for step, sig in zip(dx, sigmas)]
    x_reso_grid = np.array(list(itertools.product(*x_reso)))

    def pdf(x, y, z):
        return stats.norm.pdf(x, 0, sigmas[0])*\
            stats.norm.pdf(y, 0, sigmas[1])*\
            stats.norm.pdf(z, 0, sigmas[2])

    reso = pdf(x_reso_grid[:, 0],
               x_reso_grid[:, 1],
               x_reso_grid[:, 2])

    return (x_full, x_full_grid, x_reso, x_reso_grid, reso, dr)

def smeared_pdf(pdf, e, pd, mdpi, ndots=101):
    """ """
    _, x_full_grid, _, x_reso_grid, reso, dr =\
        local_resolution_grid(e, pd, mdpi, ndots, 5)
    ndots_res = int(np.cbrt(x_reso_grid.shape[0]))
    reso = reso.reshape(ndots_res, ndots_res, ndots_res, 1)
    pdfval = pdf.pdf_vars(
        x_full_grid[:, 0],
        x_full_grid[:, 1],
        x_full_grid[:, 2]).reshape(ndots, ndots, ndots, 1)
    pdfconv = signal.fftconvolve(pdfval, reso, 'same') * dr
    idx = (ndots + 1) // 2
    return pdfconv[idx, idx, idx]
