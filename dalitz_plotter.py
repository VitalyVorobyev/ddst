""" """

import sys
import matplotlib.pyplot as plt

from lib.params import gs, gt
from lib.dndnpip import DnDnPip, dpi_dpi_plot, dd_plot, dpi_lo_plot, dpi_hi_plot

def dndnpip(E):
    E *= 10**-3

    # pdf = DnDnPip(gs, gt, E, [False, True])
    pdf = DnDnPip(gs, gt, E, [True, True])
    # pdf = DnDnPip(gs, gt, E, channels=[True, False])
    _, axs = plt.subplots(1, 4, figsize=(16,4))

    dpi_dpi_plot(axs[0], pdf, logplot=False)
    dd_plot(     axs[1], pdf, sqrt=True)
    dpi_lo_plot( axs[2], pdf, sqrt=True)
    dpi_hi_plot( axs[3], pdf, sqrt=True)

    plt.tight_layout()

    plt.savefig(f'plots/dp_dndnpip_{E*10**3:.1f}.png')
    plt.savefig(f'plots/dp_dndnpip_{E*10**3:.1f}.pdf')

    plt.show()

if __name__ == '__main__':
    key, E = sys.argv[1], float(sys.argv[2])
    if key == 'dndnpip':
        dndnpip(E)
