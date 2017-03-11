import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import binarize

from emfrbm.emf_rbm import EMF_RBM
from emfrbm.rbm_datasets import load_omniglot_iwae

X_train, Y_train, _, X_test, Y_test, _ = load_omniglot_iwae()
logistic = linear_model.LogisticRegression()


class EmfRBMTh(EMF_RBM):
    """ A wrapper class for optimising threshold via GridsearchCV"""

    def __init__(self, n_components=256, learning_rate=0.005, batch_size=100,
                 sigma=0.001, neq_steps=3, n_iter=20, verbose=0,
                 random_state=None, momentum=0.5, decay=0.01,
                 weight_decay='L1', thresh=1e-8, monitor=False,
                 threshhold=0.5):
        super(EmfRBMTh, self).__init__(n_components=n_components,
                                       learning_rate=learning_rate,
                                       batch_size=batch_size,
                                       sigma=sigma,
                                       neq_steps=neq_steps,
                                       n_iter=n_iter,
                                       verbose=verbose,
                                       random_state=random_state,
                                       momentum=momentum,
                                       decay=decay,
                                       weight_decay=weight_decay,
                                       thresh=thresh,
                                       monitor=monitor)
        self.threshhold = threshhold

    def fit(self, X, y=None):
        X_t = binarize(X, threshold=self.threshhold, copy=True)
        return super(EmfRBMTh, self).fit(X_t, y)

    def transform(self, X):
        X_t = binarize(X, threshold=self.threshhold, copy=True)
        return super(EmfRBMTh, self).transform(X_t)

emf_rbm = EmfRBMTh(verbose=True, monitor=True)

classifier = Pipeline(steps=[('rbm', emf_rbm), ('logistic', logistic)])

param_dict = {'rbm__n_iter': [50, 500],
              'rbm__learning_rate': [0.001, 0.01, 0.1, 1],
              'rbm__decay': [0.05, 0.1, 0.2],
              'rbm__sigma': [0.001, 0.01, 0.1],
              'rbm__threshhold': [0.25, 0.5, 0.75],
              'logistic__C': [0.01, 1.0, 1.0e2, 1.0e4]}

estimator = GridSearchCV(classifier,
                         param_dict,
                         n_jobs=6,
                         iid=False,
                         pre_dispatch='2*n_jobs',
                         verbose=True,
                         cv=3)

estimator.fit(X=X_train, y=Y_train)

pd.DataFrame.from_dict(
    estimator.cv_results_).sort_values(by='rank_test_score').to_csv(
    'emf_rbm.csv')
