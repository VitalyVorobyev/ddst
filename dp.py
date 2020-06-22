#! /usr/bin/env python
""" Drivers for Dalitz distribution plots """

import sys
import matplotlib.pyplot as plt

from lib.params import gs, gt
from lib.dndnpip import DnDnPip
from lib.dndpgam import DnDpGam
from lib.dndppin import DnDpPin
from lib.plots import *

def dndpgam(E):
    """ D0 D+ gamma Dalitz plot and projections """
    E *= 1e-3

    pdf = DnDpGam(gs, gt, E)
    fig, axs = plt.subplots(2, 4, figsize=(16,8))

    dga_dga_plot(axs[0,0], pdf)
    dd_dga_plot(axs[1,0], pdf)
    dd_plot(axs[0,1], pdf, False)
    dd_plot(axs[1,1], pdf, True)
    dnga_plot(axs[0,2], pdf, False)
    dnga_plot(axs[1,2], pdf, True)
    dpga_plot(axs[0,3], pdf, False)
    dpga_plot(axs[1,3], pdf, True)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/dp_dndpgam_{E*10**3:.1f}.{ext}')

    plt.show()


def dndppin(E):
    """ D0 D+ pi0 Dalitz plot and projections """
    E *= 1e-3

    # pdf = DnDpPin(gs, gt, E, channels=[False, False, True])
    pdf = DnDpPin(gs, gt, E, channels=[True, True, True])
    _, axs = plt.subplots(1, 4, figsize=(16,4))

    logplot = True
    dpi_dpi_plot(axs[0], pdf, logplot=logplot)
    dd_plot(  axs[1], pdf, True)
    dnpi_plot(axs[2], pdf, True)
    dppi_plot(axs[3], pdf, True)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/dp_dndppin_{E*10**3:.1f}.{ext}')

    plt.show()


def dndnpip(E):
    """ D0 D0 pi+ Dalitz plot and projections """
    E *= 1e-3

    # pdf = DnDnPip(gs, gt, E, [False, True])
    pdf = DnDnPip(gs, gt, E, [True, True])
    # pdf = DnDnPip(gs, gt, E, channels=[True, False])
    _, axs = plt.subplots(1, 4, figsize=(16,4))

    dpi_dpi_plot(axs[0], pdf, logplot=False)
    dd_plot(     axs[1], pdf, sqrt=True)
    dpi_lo_plot( axs[2], pdf, sqrt=True)
    dpi_hi_plot( axs[3], pdf, sqrt=True)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'plots/dp_dndnpip_{E*10**3:.1f}.{ext}')

    plt.show()

if __name__ == '__main__':
    try:
        key, E = sys.argv[1], float(sys.argv[2])
    except (ValueError, IndexError):
        print(\
'Usage: python dalitz_plotter.py [key] [E]\n\
    - key is one of {ddpip, ddpin, ddgam}\n\
    - E (MeV) is energy relative to threshold')
        exit(0)

    if key == 'ddpip':
        dndnpip(E)
    elif key == 'ddpin':
        dndppin(E)
    elif key == 'ddgam':
        dndpgam(E)
    else:
        print(f'Wrong key: {key}')
