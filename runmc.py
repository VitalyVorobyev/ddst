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

def make_space(erng, tddmax, mdpirng):
    """ """
    return (
        ((erng[0]*10**-3, erng[1]*10**-3)), # energy range
        ((2*mdn)**2, (2*mdn + tddmax*10**-3)**2),  # m(DD) range
        (mdpirng[0]**2, mdpirng[1]**2),  # m(Dpi+) range
    )


def gen_ddpip(energy_range, tdd_max, mdpi_range, chunks):
    """ """
    space = make_space(energy_range, tdd_max, mdpi_range)

    pdf = DnDnPip(gs, gt)
    pdfa = lambda x: pdf.pdf3d(x[:,0], x[:,1], x[:,2])

    driver = MCProducer(pdfa, space)
    data = driver(chunks=chunks)
    data[:,0] *= 10**3

    np.save('mc_ddpip', data)


def gen_energy(chunks):
    """ """
    energy_space = ((-3.e-3, 15.e-3),)  # comma _is_ important

    tmtx = TMtx(gs, gt)
    pdf = lambda x: MagSq(np.sum(tmtx.vec(x)[0], axis=0))
    
    driver = MCProducer(pdf, energy_space)
    data = driver(chunks=chunks)

    print(data.shape)

    plt.hist(data, bins=150)
    plt.show()

    np.save('mc_ddpip_tmtx', data)


def phsp_plot(energy_range, tdd_max, mdpi_range):
    """ """
    space = make_space(energy_range, tdd_max, mdpi_range)

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
        energy_range = [-2.5, 15]
        tdd_max = 22
        mdpi_range = [2.004, 2.026]

        phsp_plot(energy_range, tdd_max, mdpi_range)
        chunks = int(sys.argv[1])
        gen_ddpip(energy_range, tdd_max, mdpi_range, chunks=chunks)
    except IndexError:
        print(f'Usage: ./runtoymc [chunks]')
