import sys
import numpy as np
from sklearn.datasets import load_digits
from scipy.sparse import csr_matrix
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


def test_one_batch():
    """
    Test one batch exactly
    sigma ~ 0, decay = 0 (no regularization)
    note: julia code must also use 1e-8 as eps, not 1e-6
    norm of W 0.007628825568441182
    norm W2: 1.8196753262900838e-6

    hb 1.004859173557616e-18
    vb 38.97452180357592

    pseudo l-hood: -12.962946404073032
    entropy: 68.52758766764042
    TAP free_energy: -24.38378040255592
    U naive: -48.5436547091488
    free energy: -9268.746331749979
    """
    X = Xdigits.copy()
    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=100,
                  decay=0.00, learning_rate=0.005, n_iter=0,
                  sigma=1e-16, neq_steps=3, verbose=True, weight_decay=None)
    rbm.init_weights(X)

    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.0)
    assert_almost_equal(np.linalg.norm(rbm.W2, ord=2), 0.0)
    assert_almost_equal(np.linalg.norm(rbm.dW_prev, ord=2), 0.0)

    X_batch = X[0:100, :]
    assert_almost_equal(np.linalg.norm(X_batch, ord=2), 49.31032989212045)

    rbm.partial_fit(X_batch)
    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.007628825568441182)

    scored_free_energy = np.average(rbm.score_samples(X_batch))
    avg_free_energy_tap = np.average(rbm._free_energy_TAP(X_batch))
    avg_entropy = np.average(np.average(rbm._entropy(X_batch)))

    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.9745218036)
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2), 0.0)
    assert_almost_equal(np.linalg.norm(rbm.dW_prev, ord=2), 152.57651136882203)
    assert_almost_equal(np.linalg.norm(rbm.W2, ord=2), 0.000001819675326290)

    assert_almost_equal(avg_entropy, 68.52758766764042, decimal=12)
#    assert_almost_equal(avg_free_energy_tap, -24.383780402555928)

    return rbm


def test_two_batches():
    """
    Test 2 batches exactly
    """
    X = Xdigits.copy()
    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=100,
                  decay=0.00, learning_rate=0.005, n_iter=0,
                  sigma=1e-16, neq_steps=3, verbose=True, weight_decay=None)
    rbm.init_weights(X)
    X_batch = X[0:100, :]
    rbm.partial_fit(X_batch)
    X_batch = X[100:200, :]
    rbm.partial_fit(X_batch)

    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.015478158879359825)

    scored_free_energy = np.average(rbm.score_samples(X_batch))
    avg_free_energy_tap = np.average(rbm._free_energy_TAP(X_batch))
    avg_entropy = np.average(np.average(rbm._entropy(X_batch)))

    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.974504602139554)
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2),
                        1.0779652694386856e-6)
    assert_almost_equal(np.linalg.norm(rbm.dW_prev, ord=2), 178.06423558738115)
    assert_almost_equal(np.linalg.norm(rbm.W2, ord=2), 8.120675004806954e-6)
    # assert_almost_equal(avg_entropy, )
    # assert_almost_equal(avg_free_energy_tap, )

    return rbm


def test_one_epoch():
    """
    Test one epoch, sigma=0.001
    compare:
    5 julia runs
    batch norm of W, hb, vb 0.015177951725370209 6.125160958113443e-5 38.974531344645186
    batch norm of W, hb, vb 0.016005072745766846 6.132506125735679e-5 38.974534343561935
    batch norm of W, hb, vb 0.015518275427920199 6.143705375221393e-5 38.97453267232916 batch norm of W, hb, vb 0.016618832753491925 6.14604623830071e-5 38.97453303623846 batch norm of W, hb, vb 0.015643733669880935 6.131198883353152e-5 38.97453109464897

    10 BernoulliRBM runs
    """
    X = Xdigits.copy()
    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=100,
                  decay=0.01, learning_rate=0.005, n_iter=1,
                  sigma=0.001, neq_steps=3, verbose=False)
    rbm.fit(X)

    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.974531,
                        decimal=4)
    # really between 0.015 and 0.0165: hard to test properly with a single statement

    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.0165, decimal=2)
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2), 0.000061,
                        decimal=2)

    # non tap FE totally wrong
    # FE ~ -2x.x

    scored_free_energy = np.average(rbm.score_samples(X))

    avg_free_energy_tap = np.average(rbm._free_energy_TAP(X))
    avg_entropy = np.average(np.average(rbm._entropy(X)))

    # assert_almost_equal(scored_free_energy, -24, decimal=0)
    # assert_almost_equal(avg_free_energy_tap, -25, decimal=0)
    assert_almost_equal(avg_entropy, 68.8, decimal=0)


def test_partial_fit():
    X = Xdigits.copy()
    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=100,
                  decay=0.01, learning_rate=0.005, n_iter=0,
                  sigma=0.000000001, neq_steps=3, verbose=True)
    rbm.init_weights(X)
    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.9745518)
    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.000000001)
    assert_almost_equal(np.linalg.norm(rbm.W2, ord=2), 0.000000001)
    assert_almost_equal(np.linalg.norm(rbm.dW_prev, ord=2), 0.000000001)
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2), 0.000000001)

    X_batch = Xdigits.copy()[0:100]
    assert_almost_equal(np.linalg.norm(X_batch, ord=2), 49.3103298921)
    rbm.partial_fit(X_batch)
    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.007629, decimal=4)
    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.974521,
                        decimal=4)
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2), 0.0, decimal=3)

    # there are large variations in dw_prev
    assert_almost_equal(np.linalg.norm(rbm.dW_prev, ord=2), 152.6, decimal=1)
    # TODO: make a stochastic assertion based on 100 runs
    # test stochastically (sometimes will fail due to roundoff error in dw_prev)
    # for i in range(100):
    #     test_partial_fit()


def test_fit_xdigits():
    X = Xdigits.copy()
    rbm = EMF_RBM(momentum=0.5, n_components=64, batch_size=100,
                  decay=0.01, learning_rate=0.005, n_iter=20,
                  sigma=0.001, neq_steps=3, verbose=False)
    rbm.fit(X)

    assert_almost_equal(np.linalg.norm(rbm.W, ord=2), 0.02, decimal=1)
    assert_almost_equal(np.linalg.norm(rbm.v_bias, ord=2), 38.9747, decimal=3)
    # why is h so different ?
    assert_almost_equal(np.linalg.norm(rbm.h_bias, ord=2), 0.0012, decimal=2)


def test_sample_hiddens():
    rng = np.random.RandomState(0)
    X = Xdigits[:100]
    rbm1 = EMF_RBM(n_components=2, batch_size=5, n_iter=5, random_state=42)
    rbm1.fit(X)

    h = rbm1._mean_hiddens(X[0])
    hs = np.mean([rbm1._sample_hiddens(X[0]) for i in range(100)], 0)

    assert_almost_equal(h, hs, decimal=1)


def test_rbm_verbose():
    """
    What are we testing here?
    """
    from sklearn.externals.six.moves import cStringIO as StringIO
    rbm = EMF_RBM(n_iter=2, verbose=10)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        rbm.fit(Xdigits)
    finally:
        sys.stdout = old_stdout


def test_transform():
    # using 100 causes divide by zero error in mean_hiddens()!
    X = Xdigits[:110]
    rbm1 = EMF_RBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    rbm1.fit(X)

    Xt1 = rbm1.transform(X)
    Xt2 = rbm1._mean_hiddens(X)

    assert_array_equal(Xt1, Xt2)


def test_mean_hiddens():
    # Im not entirely sure why this happens, but the hidden units all go to 1/2
    # and the h array is (2,2)
    # need to do by hand
    rng = np.random.RandomState(42)
    X = np.array([[0.], [1.]])
    rbm = EMF_RBM(n_components=2, batch_size=2,
                        n_iter=42, random_state=rng,
                        decay = 0.0, weight_decay=None, momentum=0)
    rbm.fit(X)
    h = rbm._mean_hiddens(X)
    assert h.shape == (2, 2)
    assert_almost_equal(np.linalg.norm(h, ord=2), 1.0, decimal=4)
    assert_almost_equal(h[0, 0], 0.5, decimal=3)
    assert_almost_equal(h[0, 1], 0.5, decimal=3)
    assert_almost_equal(h[1, 0], 0.5, decimal=3)
    assert_almost_equal(h[1, 1], 0.5, decimal=3)


def test_fit_equilibrate():
    # Equlibrate on the RBM hidden layer should be able to recreate [[0], [1]]
    # from the same input
    rng = np.random.RandomState(42)
    X = np.array([[0.], [1.]])
    rbm1 = EMF_RBM(n_components=2, batch_size=2,
                        n_iter=42, random_state=rng)
    # you need that much iters
    rbm1.fit(X)
    #assert_almost_equal(rbm1.W, np.array([[0.02649814], [0.02009084]]), decimal=4)
    #assert_almost_equal(rbm1.gibbs(X), X)
    return rbm1, X


def test_small_sparse():
    """
    Test using sparse CSR matrix. Need to check sparse matrix results.
    Must confirm that sparse matrix multiplies are not mis-interpeted as
    matrix dot products.
    Just testing functionality
    """
    # EMF_RBM should work on small sparse matrices.
    X = csr_matrix(Xdigits[:4])
    EMF_RBM().fit(X)       # no exception


def test_free_energy():
    """
    Test Free Energy and Entropy Calculations
    Actual julia output from xdigits_FE.jl

    Info statements added into monitor.
    I should find a way to add to test code

    S [68.92548157440935,68.67917042062827,68.7382937888165,68.6467445638933,
        68.94092201361534]
    FE <1-5>[-90.41392263605844 -98.57232874119751
        -96.67160538171822 -99.72457836849503 -89.84668949506056]
    FE TAP <1-5>[-117.38187025746029 -117.39051762052955
        -117.39024519247155 -117.39408456128287 -117.37959261213285]

    ?? m vis, hid 11.117791807149395 8.936583879865992
    ?? denoised m vis, hid 11.11779180715007 8.936583879865992
    """
    X = Xdigits.copy()
    rbm = EMF_RBM(n_iter=1, n_components=64, decay=0.001,
                  sigma=0.0000000000000001, neq_steps=5)
    rbm.fit(X)

    s = rbm._entropy(X)
    print "entropy ", s[0:5]

    fe = rbm._free_energy(X)
    print "free energies, old ", fe[0:5]

    fe_tap = rbm._free_energy_TAP(X)
    print "TAP free energies ", fe_tap[0:5]

    assert_almost_equal(s[0], 68.92548, decimal=3)
    assert_almost_equal(s[1], 68.679170, decimal=3)  # a bit more off

    assert_almost_equal(fe[0], -90.4139, decimal=3)
    assert_almost_equal(fe[1], -98.5723, decimal=2)  # a bit more off

    # assert_almost_equal(fe_tap[0], -117.3819, decimal=2)
    # assert_almost_equal(fe_tap[1], -117.3905, decimal=2)

