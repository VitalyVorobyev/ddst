""" Tools for local numerical convolution with resolution """

import itertools
import numpy as np
from scipy import stats, signal

from . import resolution as res
from .params import mdn, mpip, mdstp


def get_resolution(e, pd):
    """  """
    return (
        res.smddpi2(e / 10**3, pd / 10**3)*10**3,
        res.spd()*10**3, 
        res.smdstp()*10**3
    )

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
        local_resolution_grid(e, pd, mdpi, ndots, 7, 5)
    ndots_res = int(np.cbrt(x_reso_grid.shape[0]))
    reso = reso.reshape(ndots_res, ndots_res, ndots_res, 1)
    pdfval = pdf.pdf_vars(
        x_full_grid[:, 0],
        x_full_grid[:, 1],
        x_full_grid[:, 2]).reshape(ndots, ndots, ndots, 1)
    pdfconv = signal.fftconvolve(pdfval, reso, 'same') * dr
    idx = (ndots + 1) // 2
    return pdfconv[idx, idx, idx]
