#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py

with h5py.File('mnist.h5', 'r') as hf:
    print('List of arrays in this file: \n', hf.keys())

    b = np.array(hf.get('mnist___bias'))
    vb = np.array(hf.get('mnist___vbias'))
    w = np.array(hf.get('mnist___weight'))
   
    print(b.shape, vb.shape, w.shape)
