#! /usr/bin/env python

""" """

import numpy as np
import matplotlib.pyplot as plt

from lib.dndnpip import DnDnPip
from lib.convolution import meshgrid
from lib.plots import draw_pdf_projections

def run(ranges, gridsize=100):
    """ """
    ticks = [np.linspace(lo, hi, gridsize) for lo, hi in ranges]
    grid = meshgrid(ticks).reshape(gridsize, gridsize, gridsize, 3)
    print(f'Grid shape {grid.shape}')

    pdf = DnDnPip()
    f1 = pdf.pdf_vars(grid[:,:,:,0], grid[:,:,:,1], grid[:,:,:,2])

    fig, ax = plt.subplots(ncols=3, figsize=(18, 5.5))
    draw_pdf_projections(ax, ticks, f1)
    fig.tight_layout()

def main():
    ranges = [
        [-3., 10.],   #  energy
        [0., 150.],   #  D momentum
        [2005, 2015], #  m(Dpi)
    ]
    run(ranges=ranges, gridsize=350)
    plt.show()

if __name__ == '__main__':
    main()
