#! /usr/bin/env python

""" """

import sys
import numpy as np

from lib.params import gs, gt, mdn
from lib.toymc import MCProducer
from lib.dndnpip import DnDnPip
from lib.lineshape import TMtx, MagSq


def make_space(erng, tddmax, mdpirng):
    """ """
    return (
        (erng[0]*10**-3, erng[1]*10**-3),  # energy range
        ((2*mdn)**2, (2*mdn + tddmax*10**-3)**2),  # m(DD) range
        (mdpirng[0]**2, mdpirng[1]**2),  # m(Dpi+) range
    )


def gen_ddpip(chunks):
    """ """
    energy_range = [-3., 15.]
    tdd_max = 20
    mdpi_range = [2.004, 2.020]

    space = make_space(energy_range, tdd_max, mdpi_range)

    pdf = DnDnPip(gs, gt)
    pdf_adapter = lambda x: pdf.pdf3d(x[:,0], x[:,1], x[:,2])

    driver = MCProducer(pdf_adapter, space)
    data = driver(chunks=chunks)

    np.save('mc_ddpip', data)


def gen_energy(chunks):
    """ """
    energy_space = ((-3.e-3, 15.e-3),)  # comma IS important

    tmtx = TMtx(gs, gt)
    pdf = lambda x: MagSq(np.sum(tmtx.vec(x)[0], axis=0))
    
    driver = MCProducer(pdf, energy_space)
    data = driver(chunks=chunks)

    np.save('mc_ddpip_tmtx', data)

def plot_tmtx():
    """ """
    import matplotlib.pyplot as plt

    tmtx = TMtx(gs, gt)
    pdf = lambda x: MagSq(np.sum(tmtx.vec(x)[0], axis=0))

    E = np.linspace(-3.e-3, 15.e-3, 1000).reshape(-1,1)

    data = E[0] + np.random.random((10000, 1)) * (E[-1]-E[0])
    print(data)

    # plt.plot(E, pdf(E))
    plt.scatter(data, pdf(data), s=0.1)
    plt.show()

if __name__ == '__main__':
    try:
        plot_tmtx()
        # chunks = int(sys.argv[1])
        # gen_ddpip(chunks=chunks)
        # gen_energy(chunks=chunks)
    except IndexError:
        print(f'Usage: ./runtoymc [chunks]')
