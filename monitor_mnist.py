#!/usr/bin/env python
import h5py

from emfrbm.emf_rbm import *

hf =  h5py.File('mnist.h5', 'r')
X = np.array(hf.get('HDF5.name___X'))
y = np.array(hf.get('HDF5.name___y'))
hf.close()

emf_rbm = EMF_RBM(verbose=True, monitor=True)
print emf_rbm
emf_rbm = emf_rbm.fit(X)



