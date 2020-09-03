#! /usr/bin/env python

""" """

import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from lib.params import gs, gt, mdn, mdstp, mpip
from lib.toymc import MCProducer
from lib.dndnpip import DnDnPip
from lib.lineshape import TMtx, MagSq
from lib.dalitzphsp import Kibble


def make_espace(erng):
    """ """
    return ((erng[0]*10**-3, erng[1]*10**-3)) # energy range


def make_space_3d(erng, tddmax, mdpirng):
    """ """
    return (
        make_espace(erng),
        ((2*mdn)**2, (2*mdn + tddmax*10**-3)**2),  # m(DD) range
        (mdpirng[0]**2, mdpirng[1]**2),  # m(Dpi+) range
    )


def make_space_5d(erng, tddmax, mdpirng, gsre, gsim):
    """ """
    return (
        *make_space_3d(erng, tddmax, mdpirng),
        ((gsre[0]*10**-3, gsre[1]*10**-3)), # gsre range
        ((gsim[0]*10**-3, gsim[1]*10**-3)), # gsim range
    )

def gen_ddpip(space, chunks, igs=gs, igt=gt):
    """ Main MC driver """
    pdf = DnDnPip(igs, igt)

    driver = MCProducer(pdf.pdf, space)
    data = driver(chunks=chunks)
    data[:,0] *= 10**3
    if data.shape[1] == 5:
        data[:,3:] *= 10**3

    np.save(f'mc_ddpip_{data.shape[1]}d_gs{igs.real*10**3:.2f}_{igs.imag*10**3:.2f}_ch{chunks}', data)

def gen_ddpip_3d(ranges, chunks, igs=gs, igt=gt):
    """ """
    gen_ddpip(make_space_3d(*ranges[:3]), chunks, igs, igt)


def gen_ddpip_5d(ranges, chunks):
    """ """
    gen_ddpip(make_space_5d(*ranges), chunks)


def phsp_plot(ranges):
    """ """
    space = make_space_3d(*ranges[:3])

    x = np.linspace(*space[1], 1000)
    y = np.linspace(*space[2], 1000)
    xv, yv = np.meshgrid(x, y)
    s = (space[0][1] + mdn + mdstp)**2

    m1sq = m2sq = mdn**2
    m3sq = mpip**2

    z = Kibble(s, xv, yv, m1sq, m2sq, m3sq)

    plt.contourf(-yv-xv, yv, z)
    plt.show()


def generate_3d_on_grid(chunks):
    """ """
    # gsre = np.array([35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]) + 0.5
    # gsim = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]) + 0.05

    gsre = np.array([40., 41., 42.])
    gsim = np.array([1.5, 1.6, 1.7])

    gsre *= 10**-3
    gsim *= 10**-3

    for re, im in product(gsre, gsim):
        print(f'generating {re*10**3:.2f} + i{im*10**3:.2f} ...')
        gen_ddpip_3d(ranges, chunks=chunks,
            igs=(re+1j*im),
            igt=(25+1j*im),
        )

if __name__ == '__main__':
    try:
        ranges = (
            [-2.5, 10],  # energy
            10,  # Tdd max
            [2.005, 2.016],  # m(D0 pi+)
            [30., 40.],  # gs.real
            [1.3, 1.7],   # gs.imag
        )

        phsp_plot(ranges)
        chunks = int(sys.argv[1])
        if len(sys.argv) == 4:
            gsre = float(sys.argv[2]) * 10**-3
            gsim = float(sys.argv[3]) * 10**-3
        else:
            gsre, gsim = gs.real, gs.imag

        generate_3d_on_grid(chunks=chunks)
        # gen_ddpip_3d(ranges, chunks=chunks,
        #     igs=(gsre + 1j*gsim),
        #     igt=( 25. + 1j*gsim)
        # )

        # gen_ddpip_5d(ranges, chunks=chunks)

    except IndexError:
        print(f'Usage: ./runtoymc [chunks] [[real(gs)] [imag(gs)]]')
