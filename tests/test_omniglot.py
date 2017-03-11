import numpy as np
import pytest
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import binarize

from emfrbm.emf_rbm import EMF_RBM
from emfrbm.rbm_datasets import load_omniglot_iwae

np.seterr(all='warn')


@pytest.fixture(params=[0.25, 0.4, 0.5])
def threshold(request):
    return request.param


def test_omniglot_iwae(threshold=0.4):
    print('testing omniglot with threshold: {}'.format(threshold))
    X_train, Y_train, _, X_test, Y_test, _ = load_omniglot_iwae()
    X_train = binarize(X_train, threshold=threshold, copy=True)
    X_test = binarize(X_test, threshold=threshold, copy=True)

    X = np.vstack((X_train, X_test))
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
    X_train = (X_train - np.min(X, 0)) / (
        np.max(X, 0) + 0.0001)  # 0-1 scaling
    X_test = (X_test - np.min(X, 0)) / (
        np.max(X, 0) + 0.0001)  # 0-1 scaling

    B_rbm = BernoulliRBM(verbose=True, n_iter=20)
    logistic = linear_model.LogisticRegression()
    logistic.C = 5000

    classifier = Pipeline(steps=[('rbm', B_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)

    Y_test_berboulli_pred = classifier.predict(X_test)

    # one iteration of emb rbm on my PC is is about 50% faster
    # about 8 secs compared to 12 secs for Bernoulli RBM. So I train emf-rbm
    # for 50% more iterations
    emf_rbm = EMF_RBM(verbose=True, n_iter=30)

    classifier = Pipeline(steps=[('rbm', emf_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)
    Y_test_emf_pred = classifier.predict(X_test)
    emf_accuracy = accuracy_score(y_pred=Y_test_emf_pred, y_true=Y_test)
    bernoulli_accuracy = accuracy_score(y_pred=Y_test_berboulli_pred,
                                        y_true=Y_test)
    print(emf_accuracy, bernoulli_accuracy)
    assert abs(emf_accuracy - bernoulli_accuracy) < 0.1
