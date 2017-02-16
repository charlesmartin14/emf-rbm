import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils.validation import assert_all_finite
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.preprocessing import Binarizer
from emf_rbm import EMF_RBM
np.seterr(all='warn')

Xdigits = load_digits().data
Xdigits -= Xdigits.min()
Xdigits /= Xdigits.max()

b = Binarizer(threshold=0.001, copy=True)
Xdigits = b.fit_transform(Xdigits)


def test_init():
    X = Xdigits.copy()
    assert_almost_equal(np.linalg.norm(X, ord=2), 211.4983270228649,
                        decimal=12)

    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=50 , decay=0.01,
                  learning_rate=0.005, n_iter=0, sigma=0.001, neq_steps=3,
                  verbose=True)
    rbm.fit(X)
    assert np.linalg.norm(rbm.h_bias, ord=2) == 0.0
    assert np.linalg.norm(rbm.lr) == 0.005
    assert np.linalg.norm(rbm.momentum) == 0.5
    assert np.linalg.norm(rbm.decay) == 0.01
    assert np.linalg.norm(rbm.n_iter) == 0
    assert np.linalg.norm(rbm.neq_steps) == 3
    assert np.linalg.norm(rbm.sigma)== 0.001
    assert np.linalg.norm(rbm.verbose)
    assert np.linalg.norm(rbm.n_components)
    assert np.linalg.norm(rbm.thresh) == 1e-8
    assert np.linalg.norm(rbm.batch_size) == 50

    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2),
                                   38.97455, decimal=5)
    #assert_true(np.linalg.norm(rbm.weight_decay)=='L1')
    assert_array_equal(X, Xdigits)
