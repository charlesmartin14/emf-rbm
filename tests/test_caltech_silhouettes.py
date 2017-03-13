import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neural_network import BernoulliRBM

from emf_rbm import EMF_RBM
from rbm_datasets import load_caltech_silhouettes
np.seterr(all='warn')


def test_caltech_silhouettes():
    X_train, Y_train, _, _, X_test, Y_test = load_caltech_silhouettes()
    X = np.vstack((X_train, X_test))
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
    X_train = (X_train - np.min(X, 0)) / (
        np.max(X, 0) + 0.0001)  # 0-1 scaling
    X_test = (X_test - np.min(X, 0)) / (
        np.max(X, 0) + 0.0001)  # 0-1 scaling

    B_rbm = BernoulliRBM(verbose=True, n_iter=10)
    logistic = linear_model.LogisticRegression()
    logistic.C = 5000

    classifier = Pipeline(steps=[('rbm', B_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)

    Y_test_berboulli_pred = classifier.predict(X_test)

    emf_rbm = EMF_RBM(verbose=True, n_iter=10)

    classifier = Pipeline(steps=[('rbm', emf_rbm), ('logistic', logistic)])
    classifier.fit(X_train, Y_train)
    Y_test_emf_pred = classifier.predict(X_test)
    emf_accuracy = accuracy_score(y_pred=Y_test_emf_pred, y_true=Y_test)
    bernoulli_accuracy = accuracy_score(y_pred=Y_test_berboulli_pred,
                                        y_true=Y_test)
    assert abs(emf_accuracy - bernoulli_accuracy) < 0.1
