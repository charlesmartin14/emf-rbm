#!/usr/bin/env python
import numpy as np
import h5py

from sklearn.datasets import fetch_mldata
from sklearn.utils.validation import assert_all_finite
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sklearn.preprocessing import Binarizer

import emf_rbm
from emf_rbm import *

hf =  h5py.File('mnist.h5', 'r')
X = np.array(hf.get('HDF5.name___X'))
y = np.array(hf.get('HDF5.name___y'))
hf.close()

emf_rbm = EMF_RBM(verbose=True, monitor=True)
print emf_rbm
emf_rbm = emf_rbm.fit(X)



