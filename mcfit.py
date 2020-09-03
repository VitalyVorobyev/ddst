""" Max likelihood fit of the smeared MC sample """

import os
import numpy as np

from lib.convolution import local_resolution_grid
from lib.vartools import generated_to_observables, observables_to_mandelstam
from lib.dndnpip import DnDnPip

import lib.params as pars

path_data = './mcsamples'

def sample_fname(re, im ,ch):
    """ Smeared toy MC data set file name """
    fname = os.path.join(
        path_data, f'mc_ddpip_3d_gs{re:.2f}_{im:.2f}_ch{ch}_smeared.npy')
    if os.path.isfile(fname):
        return fname
    print(f'file {fname} not found')
    return None
    
def get_sample(re, im, ch, nevt):
    """ Load smeared toy MC data set """
    fname = sample_fname(re, im, ch)
    if fname:
        data = np.load(fname)[:nevt]
        return (data[:, 0], *generated_to_observables(data[:, 1], data[:, 2]))
    else:
        return None

