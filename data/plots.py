""" """

import sys
sys.path.append('lib')

from params import datafile

import uproot
import numpy as np
import matplotlib.pyplot as plt

treekeys = ['mass_C1c', 'mass_C2c', 'mass_2Cc', 'pid_S1', 'pid_S2']

def make_hist(data, range=None, nbins=100, density=False):
    if range is None:
        range = (np.min(data), np.max(data))
    dhist, dbins = np.histogram(data, bins=nbins, density=density, range=range)
    dbins = 0.5 * (dbins[1:] + dbins[:-1])
    norm = np.sum(dhist) / data.shape[0]
    errors = np.array([-0.5 + np.sqrt(dhist / norm + 0.25),
                        0.5 + np.sqrt(dhist / norm + 0.25)]) * norm
    return (dbins, dhist, errors)

def load_tree(infile=datafile, keys=treekeys):
    """ """
    return uproot.open(infile)['tree'].pandas.df(keys)

def md_plot(df, r=(1.8, 1.925)):
    """ """
    dbins, dhist, errors = make_hist(df['mass_C1c'], range=r)
    plt.figure(figsize=(8,6))
    plt.errorbar(dbins, dhist, errors, linestyle='none')
    plt.xlim(*r)
    plt.ylim(0, 1.05*max(dhist))
    plt.xlabel(r'$m(D^0), GeV$')
    plt.grid()
    plt.tight_layout()
    plt.show()

def mdpi_plot(df, r=(2.006, 2.016)):
    """ """
    dbins, dhist, errors = make_hist(df['mass_C2c'], range=r)
    plt.figure(figsize=(8,6))
    plt.errorbar(dbins, dhist, errors, linestyle='none')
    plt.xlim(*r)
    plt.ylim(0, 1.05*max(dhist))
    plt.xlabel(r'$m(D^{*+}), GeV$')
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    """ Unit test """
    df = load_tree()
    print(df.head())
    mdpi_plot(df)

if __name__ == '__main__':
    main()
