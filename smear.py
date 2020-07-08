#! /usr/bin/env python

import sys
# import jax.numpy as np
import numpy as np
from lib.resolution import sample

def smear(ifname : str, ofname : str =None) -> (None):
    """ Applies resolution to MC events """
    data = np.array(np.load(ifname))
    if ofname is None:
        ofname = f'{ifname[:-4]}_smeared'
    
    sdata = sample(data)
    
    print(f'E std: {np.std(sdata[:,0] - data[:,0])}')

    np.save(ofname, sdata)


if __name__ == '__main__':
    try:
        ifname = sys.argv[1]
        if not ifname.endswith('.npy'):
            raise ValueError
        smear(ifname)
    except (ValueError,IndexError):
        print('Usage: ./smear.py [input file].npy')
