#!/usr/bin/env python
import numpy as np
import h5py

with h5py.File('mnist.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
