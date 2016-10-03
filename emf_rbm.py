import time

import numpy as np
import scipy.sparse as sp


from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.validation import check_is_fitted

from sklearn.utils.fixes import expit  # logistic function  
from sklearn.utils.extmath import safe_sparse_dot, log_logistic

class EMF_RBM(BaseEstimator, TransformerMixin):
    """Extended Mean Field Restricted Boltzmann Machine (RBM).
    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using the Extended Mean
    Field model, based on the TAP equations
    Read more in the :ref:`User Guide <rbm>`.
    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.
    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.
    batch_size : int, optional
        Number of examples per minibatch.
    momentum : float, optional
        gradient momentum parameter
    decay : float, optional
        decay for weight update regularizer
    weight_decay: string, optional []'L1', 'L2', None]
        weight update regularizer

    neq_steps: int, optional
        Number of equilibration steps
    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.
    sigma: float, optional
        variance of initial W weight matrix
    thresh: float, optional
        threshold for values in W weight matrix, vectors
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is give it fixes the
        seed. Defaults to the global numpy random number generator.
    Attributes
    ----------
    h_bias : array-like, shape (n_components,)
        Biases of the hidden units.
    v_bias : array-like, shape (n_features,)
        Biases of the visible units.
    W : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = EMF_RBM(n_components=2)
    >>> model.fit(X)
    EmfRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=0)
    References
    ----------
    [1] Marylou Gabrie´, Eric W. Tramel1 and Florent Krzakala1, 
        Training Restricted Boltzmann Machines via the Thouless-Anderson-Palmer Free Energy
        https://arxiv.org/pdf/1506.02914
    """
    def __init__(self, n_components=256, learning_rate=0.005, batch_size=100, sigma=0.001, neq_steps = 3,
                 n_iter=20, verbose=0, random_state=None, momentum = 0.5, decay = 0.01, weight_decay='L1', thresh=1e-8):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose

        self.momentum = momentum
        self.decay = decay
        self.weight_decay = weight_decay

        self.sigma = sigma
        self.neq_steps = neq_steps

        # learning rate / mini_batch
        self.lr = learning_rate

        # threshold for floats
        self.thresh = thresh

        # store in case we want to reset
        self.random_state = random_state
        

        # self.random_state_ = random_state
        # always start with new random state
        self.random_state = check_random_state(random_state)


        # initialize arrays to 0
        self.W = np.asarray(
            self.random_state.normal(
                0,
                sigma,
                (self.n_components, X.shape[1])
            ),
            order='fortran')

        self.dW_prev = np.zeros_like(self.W)
        self.W2 = self.W*self.W

        self.h_bias = np.zeros(self.n_components, )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        self.v_bias = self.init_v_bias(X)

    def init_v_bias(self, X):
        """ If the user specifies the training dataset, it can be useful to                                                                                   
        initialize the visibile biases according to the empirical expected                                                                                
        feature values of the training data.                                                                                                              

        TODO: Generalize this biasing. Currently, the biasing is only written for                                                                         
               the case of binary RBMs.
        """
        # 
        eps = self.thresh

        probVis = np.mean(X,axis=0)             # Mean across  samples 
        probVis[probVis < eps] = eps            # Some regularization (avoid Inf/NaN)  
        #probVis[probVis < (1.0-eps)] = (1.0-eps)   
        v_bias = np.log(probVis / (1.0-probVis)) # Biasing as the log-proportion  
        return v_bias


    def sample_layer(self, layer):
        """Sample from the conditional distribution P(h|v) or P(v|h)"""
        self.random_state = check_random_state(self.random_state)
        return (self.random_state.random_sample(size=l.shape) < layer)  

    def _sample_hiddens(self, v):
        """Sample from the conditional distribution P(h|v).
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.
        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        return sample_layer(self._mean_hiddens(v))

    def _mean_hiddens(self, v):
        """Computes the conditional probabilities P(h=1|v).
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.W.T) + self.h_bias
        return expit(p, out=p)

    def _sample_visibles(self, h):
        """Sample from the distribution P(v|h).
        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.
        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        return sample_layer(self._mean_visible(h))

    def _mean_visibles(self, h):
        """Computes the conditional probabilities P(v=1|h).
        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        Returns
        -------
         v : array-like, shape (n_samples, n_features)
            Values of the visible layer.     
        """
        p = np.dot(h, self.W) + self.v_bias
        return expit(p, out=p)

    def sigma_means(self, x, b, W):
        """helper class for computing Wx+b """
        a = safe_sparse_dot(x, W.T) + b
        return expit(a, out=a)

    def init_batch(self, vis):
        """initialize the batch for EMF only"""
        v_pos = vis
        v_init = v_pos

        h_pos = self._mean_hiddens(v_pos)
        h_init = h_pos

        return v_pos, h_pos, v_init, h_init

    def equilibrate(self, v0, h0, iters=3):
        """Run iters steps of the TAP fixed point equations"""
        mv = v0
        mh = h0
        for i in range(iters):
            mv = 0.5 *self.mv_update(mv, mh) + 0.5*mv
            mh = 0.5 *self.mh_update(mv, mh) + 0.5*mh

        return mv, mh

    def mv_update(self, v, h):  
        """update TAP visbile magnetizations, to second order"""
        a = np.dot(h, self.W) + self.v_bias

        h_fluc = h-(h*h)
        a += h_fluc.dot(self.W2)*(0.5-v)
        return expit(a, out=a)

    def mh_update(self, v, h):
        """update TAP hidden magnetizations, to second order"""
        a = safe_sparse_dot(v, self.W.T) + self.h_bias

        v_fluc = v-(v*v)
        a += v_fluc.dot(self.W2.T)*(0.5-h)
        return expit(a, out=a)


    def weight_gradient(self, v_pos, h_pos ,v_neg, h_neg):
        """compute weight gradient of the TAP Free Energy, to second order"""
        # naive  / mean field
        dW = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T - np.dot(h_neg.T, v_neg)

        # tap2 correction
        h_fluc = (h_neg - (h_neg*h_neg)).T
        v_fluc = (v_neg - (v_neg*v_neg))
        dW_tap2 = h_fluc.dot(v_fluc)*self.W

        dW -= dW_tap2
        return dW

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).
        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).
        Notes
        -----
        This method is not deterministic: it computes the TAP Free Energy on X,
        then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, \W\)

        v = check_array(X, accept_sparse='csr')
        self.random_state = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               self.random_state.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)

    #TODO: fix later
    def _denoise(m, eps=1e-8):
        """denoise magnetization"""
      #  m[m < eps] = eps
        return m


    def _free_energy(self, v):
        """Computes the TAP Free Energy F(v) to second order
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        fe = (- safe_sparse_dot(v, self.v_bias)
                - np.logaddexp(0, safe_sparse_dot(v, self.W.T)
                               + self.h_bias).sum(axis=1))
        
        h = self._mean_hiddens(v)
        mv, mh = self.equilibrate(v, h, iters=self.neq_steps)
        
        #TODO: implement / test
        #mv = self._denoise(mv)
        #mh = self._denoise(mh)

        # sum over nodes: axis=1
        print \mv mh W2 shapes\, mv.shape, mh.shape, self.W2.shape
        
        U_naive = (-safe_sparse_dot(mv, self.v_bias) 
                    -safe_sparse_dot(mh, self.h_bias) 
                        -(mv.dot(self.W.T)*(mh)).sum(axis=1))     

        Entropy = ( -(mv*np.log(mv)+(1.0-mv)*np.log(1.0-mv)).sum(axis=1)  
                    -(mh*np.log(mh)+(1.0-mh)*np.log(1.0-mh)).sum(axis=1) )
                   
        h_fluc = (mh - (mh*mh))
        v_fluc = (mv - (mv*mv))
        dW_tap2 = h_fluc.dot(self.W2).dot(v_fluc.T)
        
        print \U_naive, Entropy, dW_tap2\, U_naive.shape,Entropy.shape, dW_tap2.shape

        Onsager = - (0.5 * dW_tap2).sum(axis=1)
        

        fe_tap = U_naive + Onsager - Entropy

        return fe_tap - fe


    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : EMF_RBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'intercept_hidden_'):
            self.h_bias = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.v_bias = np.zeros(X.shape[1], )

        # not used ?
        #if not hasattr(self, 'h_samples_'):
        #    self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def _fit(self, v_pos):
        """Inner fit for one mini-batch.
        Adjust the parameters to maximize the likelihood of v using
        Extended Mean Field theory (second order TAP equations).
        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        """
        X_batch = v_pos
        lr = float(self.learning_rate) / X_batch.shape[0]
        decay = self.decay

        v_pos, h_pos, v_init, h_init = self.init_batch(X_batch)

        # get_negative_samples
        v_neg, h_neg = self.equilibrate(v_init, h_init, iters=rbm.neq_steps) 

        # basic gradient
        dW = self.weight_gradient(v_pos, h_pos ,v_neg, h_neg) 

        # regularization based on weight decay
        #  similar to momentum >
        if rbm.weight_decay == \L2\:
            dW += decay * np.sign(self.W)
        elif rbm.weight_decay == \L1\:
            dW += decay * self.W

        # can we use BLAS here ?
        # momentum
        # note:  what do we do if lr changes per step ? not ready yet
        dW += self.momentum * self.dW_prev  

        # update
        self.W += lr * dW 

        # for next iteration
        self.dW_prev =  dW  
        self.W2 = self.W*self.W

        # update bias terms
        self.h_bias += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.v_bias += lr * (np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0))

        return 0

    
    def fit(self, X, y=None):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        verbose = self.verbose
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        self.random_state = check_random_state(self.random_state)
        n_samples = X.shape[0]
        n_batches = int(np.ceil(float(n_samples) / rbm.batch_size))
        

        batch_slices = list(gen_even_slices(n_batches * rbm.batch_size,
                                            n_batches, n_samples))
        
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                print \.\,
                self._fit(X[batch_slice])

            if verbose:
                end = time.time()
                print(\[%s] Iteration %d, pseudo-likelihood = %.2f,\
                      \ time = %.2fs\
                      % (type(self).__name__, iteratio
                         self.score_samples(X).mean(), end - begin))
                begin = end

        return self

