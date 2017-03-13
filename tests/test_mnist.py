import numpy as np
import pytest
from scipy.ndimage import convolve
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from emfrbm.emf_rbm import EMF_RBM
from emfrbm.rbm_datasets import load_mnist_realval

np.seterr(all='warn')


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


@pytest.fixture()
def minst():
    # Load Data
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    Y = digits.target
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, Y_train, Y_test


@pytest.fixture(params=[5000])
def regulariser_C(request):
    return request.param


def test_digits(regulariser_C, minst):
    print('testing digists dataset')
    X_train, X_test, Y_train, Y_test = minst
    common_check(X_test, X_train, Y_test, Y_train, regulariser_C)


def common_check(X_test, X_train, Y_test, Y_train, regulariser_C):
    B_rbm = BernoulliRBM(verbose=True, n_components=100, n_iter=40,
                         learning_rate=0.06)
    logistic = linear_model.LogisticRegression()
    logistic.C = regulariser_C
    classifier = Pipeline(steps=[('rbm', B_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)
    Y_test_berboulli_pred = classifier.predict(X_test)
    emf_rbm = EMF_RBM(verbose=True, n_components=100, n_iter=40,
                      learning_rate=0.06)
    classifier = Pipeline(steps=[('rbm', emf_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)
    Y_test_emf_pred = classifier.predict(X_test)
    emf_accuracy = accuracy_score(y_pred=Y_test_emf_pred, y_true=Y_test)
    bernoulli_accuracy = accuracy_score(y_pred=Y_test_berboulli_pred,
                                        y_true=Y_test)
    print("bernoulli_accuracy:", bernoulli_accuracy)
    print("emf_accuracy:", emf_accuracy)
    assert abs(emf_accuracy - bernoulli_accuracy) < 0.1


def test_mnist_realval():
    x_train, targets_train, x_valid, targets_valid, x_test, targets_test = \
        load_mnist_realval()
    X = np.vstack((x_train, x_test))
    x_train = (x_train - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    x_test = (x_test - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    common_check(x_test, x_train, targets_test, targets_train, 5000)
