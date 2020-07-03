#! /usr/bin/env python

""" """

import sys
import numpy as np
import matplotlib.pyplot as plt

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

def gen_ddpip(space, chunks):
    """ Main MC driver """
    pdf = DnDnPip(gs, gt)

    driver = MCProducer(pdf.pdf, space)
    data = driver(chunks=chunks)
    data[:,0] *= 10**3
    if data.shape[1] == 5:
        data[:,3:] *= 10**3

    np.save(f'mc_ddpip_{data.shape[1]}d', data)

def gen_ddpip_3d(ranges, chunks):
    """ """
    gen_ddpip(make_space_3d(*ranges[:3]), chunks)


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

if __name__ == '__main__':
    try:
        ranges = (
            [-2.5, 15],  # energy
            22,  # Tdd max
            [2.004, 2.026],  # m(D0 pi+)
            [30., 40.],  # gs.real
            [1.3, 1.7],   # gs.imag
        )

        phsp_plot(ranges)
        chunks = int(sys.argv[1])
        # gen_ddpip_3d(ranges, chunks=chunks)
        gen_ddpip_5d(ranges, chunks=chunks)

    except IndexError:
        print(f'Usage: ./runtoymc [chunks]')
