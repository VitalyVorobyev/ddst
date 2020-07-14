#! /usr/bin/env python

import sys
import os
import numpy as np
from lib.resolution import sample

def fname_smeared(name, ext=False):
    """ Standard name for smeared file for a given generated file name """
    sname = f'{name[:-4]}_smeared'
    if ext:
        sname += '.npy'
    return sname

def smear(ifname: str, ofname: str =None) -> (None):
    """ Applies resolution to MC events """
    data = np.array(np.load(ifname))
    if ofname is None:
        ofname = fname_smeared(ifname)
    
    sdata = sample(data)
    
    print(f'saved to {ofname}')
    np.save(ofname, sdata)


def smear_files(path:str, forced: bool=False) -> (None):
    """ Smears all files in a directory.
        Skips file forced is False and corresponing smeared file already exists """
    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    all_files = set(filter(lambda x: x.startswith('mc_ddpip_3d_'), all_files))
    smeared_files = set(filter(lambda x: 'smeared' in x, all_files))
    raw_files = all_files - smeared_files

    print(f'{len(raw_files)} raw files and {len(smeared_files)} smeared files found {path}')
    print(raw_files.pop())
    print(smeared_files.pop())

    for idx, fname in enumerate(raw_files):
        print(f'processing {idx:3d}/{len(raw_files):3d}: {fname}')
        if not forced and fname_smeared(fname, True) in smeared_files:
            print('Skipped')
            continue

        smear(os.path.join(path, fname))


def main(ifname):
    if ifname.endswith('.npy'):
        smear(ifname)
    else:
        smear_files(ifname, forced=False)

if __name__ == '__main__':
    try:
        main(sys.argv[1])

    except IndexError:
        print('\n'.join([
            'Usage:',
            './smear.py [input file].npy',
            '  or',
            '/smear.py [input directory]'
        ]))
