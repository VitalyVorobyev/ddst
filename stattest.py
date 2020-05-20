#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 10**5
    bins = 100
    x = np.random.rand(N)
    sqrtx = np.sqrt(x)

    y = N*np.ones(bins) / bins
    z = np.linspace(0, 1, bins)

    plt.hist(sqrtx, bins=bins)
    plt.plot(np.sqrt(z), 2*y*np.sqrt(z))

    plt.show()

if __name__ == '__main__':
    main()