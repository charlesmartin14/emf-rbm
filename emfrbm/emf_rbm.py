from __future__ import print_function
import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import expit  # logistic function  
from sklearn.utils.extmath import safe_sparse_dot, log_logistic, softmax


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
        random permutations generator. If an integer is given, it fixes the
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
    [1] Marylou Gabrie, Eric W. Tramel1 and Florent Krzakala1, 
        Training Restricted Boltzmann Machines via the
        Thouless-Anderson-Palmer Free Energy
        https://arxiv.org/pdf/1506.02914
    """
    def __init__(self, n_components=256, learning_rate=0.005, batch_size=100,
                 sigma=0.001, neq_steps=3, n_iter=20, verbose=0,
                 random_state=None, momentum=0.5, decay=0.01,
                 weight_decay='L1', thresh=1e-8, monitor=False):
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
        
        # h bias
        self.h_bias = np.zeros(self.n_components, )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        # moved to fit
        
        self.W = None
        self.dW_prev = None
        self.W2 = None
        self.v_bias = None
        
        # internal monitors: i would prefer callbacks
        self.monitor = monitor
        self.entropies = []
        self.free_energies = []
        self.mean_field_energies = []
        
    def init_weights(self, X):
        """ If the user specifies the training dataset, it can be useful to                                                                                   
        initialize the visibile biases according to the empirical expected                                                                                
        feature values of the training data.                                                                                                              

        TODO: Generalize this biasing. Currently, the biasing is only written for                                                                         
               the case of binary RBMs.
        """
        # 
        eps = self.thresh

        # Mean across  samples 
        if issparse(X):
            probVis = sp.csr_matrix.mean(X, axis=0)
        else:
            probVis = np.mean(X, axis=0)

        # safe for CSR / sparse mats ?
        # do we need it if we use softmax ?
        probVis[probVis < eps] = eps # Some regularization (avoid Inf/NaN)
        # probVis[probVis < (1.0-eps)] = (1.0-eps)
        # Biasing as the log-proportion
        self.v_bias = np.log(probVis / (1.0-probVis))
        
        # (does not work)
        # self.v_bias = softmax(probVis)
        
        # initialize arrays to 0
        self.W = np.asarray(
            self.random_state.normal(
                0,
                self.sigma,
                (self.n_components, X.shape[1])
            ),
            order='fortran')

        self.dW_prev = np.zeros_like(self.W)
        self.W2 = self.W*self.W
        return 0

    def sample_layer(self, layer):
        """Sample from the conditional distribution P(h|v) or P(v|h)"""
        self.random_state = check_random_state(self.random_state)
        sample = (self.random_state.random_sample(size=layer.shape) < layer) 
        return sample

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
        return self.sample_layer(self._mean_hiddens(v))

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
        return self.sample_layer(self._mean_visible(h))

    def _mean_visible(self, h):
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
        # p = np.dot(h, self.W) + self.v_bias
        p = safe_sparse_dot(h, self.W) + self.v_bias
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
        
        # a = np.dot(h, self.W) + self.v_bias
        a = safe_sparse_dot(h, self.W) + self.v_bias

        h_fluc = h-np.multiply(h,h)
        #a += h_fluc.dot(self.W2)*(0.5-v)
        
        # 0.5-v is elementwise => dense
        if issparse(v):
            v_half = (0.5-v.todense())
        else:
             v_half = (0.5-v)
            
        a += np.multiply(safe_sparse_dot(h_fluc,self.W2), v_half)
        return expit(a, out=a)

    def mh_update(self, v, h):
        """update TAP hidden magnetizations, to second order"""
        a = safe_sparse_dot(v, self.W.T) + self.h_bias
 
        v_fluc = (v-(np.multiply(v,v)))
        #a += (v-v*v).dot((self.W2).T)*(0.5-h)
        
        if issparse(h):
            h_half = (0.5-h.to_dense())
        else:        
            h_half = (0.5-h)
            
        a += np.multiply(safe_sparse_dot(v_fluc,self.W2.T),h_half)

        return expit(a, out=a)

    def weight_gradient(self, v_pos, h_pos ,v_neg, h_neg):
        """compute weight gradient of the TAP Free Energy, to second order"""
        # naive  / mean field
        dW = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T - \
             np.dot(h_neg.T, v_neg)
        
        # tap2 correction
        #  elementwise multiplies
        h_fluc = (h_neg - np.multiply(h_neg,h_neg)).T
        v_fluc = (v_neg - np.multiply(v_neg,v_neg))
        #  dW_tap2 = h_fluc.dot(v_fluc)*self.W
        dW_tap2 = np.multiply(safe_sparse_dot(h_fluc,v_fluc),self.W)

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
        check_is_fitted(self, "W")

        v = check_array(X, accept_sparse='csr')
        v, v_ = self._corrupt_data(v)       

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)
    
    def score_samples_TAP(self, X):
        """Compute the pseudo-likelihood of X using second order TAP
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
        check_is_fitted(self, "W")

        v = check_array(X, accept_sparse='csr')      
        v, v_ = self._corrupt_data(v)       

        fe = self._free_energy_TAP(v)
        fe_ = self._free_energy_TAP(v_)
        return v.shape[1] * log_logistic(fe_ - fe)
    
    def _corrupt_data(self, v):
        self.random_state = check_random_state(self.random_state)
        """Randomly corrupt one feature in each sample in v."""
        ind = (np.arange(v.shape[0]),
               self.random_state.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]
        return v, v_
    
    def score_samples_entropy(self, X):
        """Compute the entropy of X
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).
        Returns
        -------
        entropy : array-like, shape (n_samples,)
            Value of the entropy.
        Notes
        -----
        This method is not deterministic: it computes the entropy on X,
        then on a randomly corrupted version of X, and returns the difference.
        """
        check_is_fitted(self, "W")

        v = check_array(X, accept_sparse='csr')
        v, v_ = self._corrupt_data(v)       

        s = self._entropy(v)
        s_ = self._entropy(v_)
        return v.shape[1] * (s_ - s)

    
        #TODO: run per column
    def _denoise(self, m, eps=1.0e-8):
        """denoise magnetization"""
        m = np.maximum(m,eps)
        m = np.minimum(m,1.0-eps)
        return m


    def _U_naive_TAP(self, v):
        """Computes the  Mean Field TAP Energy E(v) 
        Parameters. This is also U_Naive in the TAP FE
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        U_naive : array-like, shape (n_samples,)
            The value of the mean field component of the TAP free energy.
        """
        h = self._mean_hiddens(v)
        mv, mh = self.equilibrate(v, h, iters=self.neq_steps)
           
        mv = self._denoise(mv)
        mh = self._denoise(mh)
    
        # sum over nodes: axis=1
        
        U_naive = (-safe_sparse_dot(mv, self.v_bias) 
                    -safe_sparse_dot(mh, self.h_bias) 
                        -(mv.dot(self.W.T)*(mh)).sum(axis=1))         

        return U_naive
            
    def _free_energy_TAP(self, X):
        """Computes the TAP Free Energy F(v) to second order Parameters
        Also provides  values of components (energy, naive, Onsager term)
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        #fe = (- safe_sparse_dot(v, self.v_bias)
        #        - np.logaddexp(0, safe_sparse_dot(v, self.W.T)
        #                       + self.h_bias).sum(axis=1))
        
        v = check_array(X, accept_sparse='csr')      
            
        h = self._mean_hiddens(v)
        mv, mh = self.equilibrate(v, h, iters=self.neq_steps)
           
        mv = self._denoise(mv)
        mh = self._denoise(mh)
    
        # sum over nodes: axis=1
        
        U_naive = (-safe_sparse_dot(mv, self.v_bias) 
                    -safe_sparse_dot(mh, self.h_bias) 
                        -(mv.dot(self.W.T)*(mh)).sum(axis=1))

        Entropy = ( -(mv*np.log(mv)+(1.0-mv)*np.log(1.0-mv)).sum(axis=1)  
                    -(mh*np.log(mh)+(1.0-mh)*np.log(1.0-mh)).sum(axis=1) )
           
        h_fluc = (mh - (mh*mh))
        v_fluc = (mv - (mv*mv))

        # if we do it this way, we need to normalize by 1/batch_size 
        # which we need to obtain from the W2 matrix
        # (I think because of the double sum)
        # this is not obvious in the paper...have to be very careful here...too damn slow
        #tap_norm = 1.0/float(mv.shape[0])
        #dW_tap2 = h_fluc.dot(self.W2).dot(v_fluc.T)
        # Onsager = (-0.5*dW_tap2).sum(axis=1)*tap_norm
            
        # julia way, does not require extra norm, but maybe slower ?
        dW_tap2 = h_fluc.dot(self.W2)*v_fluc
        Onsager = (-0.5*dW_tap2).sum(axis=1)
        fe_tap = U_naive + Onsager - Entropy

        return fe_tap, [Entropy, U_naive, Onsager]

    def _free_energy(self, v):
        """Computes the RBM Free Energy F(v) Parameters.  
        (No mean field h values necessary)
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

        return fe

    def _entropy(self, v):
        """Computes the TAP Entropy (S) , from an equilibration step
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        entropy : array-like, shape (n_samples,)
            The value of the entropy.
        """
         
        h = self._mean_hiddens(v)
        mv, mh = self.equilibrate(v, h, iters=self.neq_steps)

        mv = self._denoise(mv)
        mh = self._denoise(mh)

        # appears to be wrong ?  unsure why ?  maybe because it is not denoised !!!
        Entropy = ( -(mv*np.log(mv)+(1.0-mv)*np.log(1.0-mv)).sum(axis=1)  
                    -(mh*np.log(mh)+(1.0-mh)*np.log(1.0-mh)).sum(axis=1)  )
                         
        return Entropy

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
        if not hasattr(self, 'W'):
            self.W = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'h_bias'):
            self.h_bias = np.zeros(self.n_components, )
        if not hasattr(self, 'v_bias'):
            self.v_bias = np.zeros(X.shape[1], )

        # not used ?
        #if not hasattr(self, 'h_samples_'):
        #    self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X)

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
              
        a = safe_sparse_dot(h_init, self.W, dense_output=True) + self.v_bias
        a = expit(a, out=a)

        # get_negative_samples
        v_neg, h_neg = self.equilibrate(v_init, h_init, iters=self.neq_steps) 
        
        # basic gradient
        dW = self.weight_gradient(v_pos, h_pos ,v_neg, h_neg) 

        # regularization based on weight decay
        #  similar to momentum >
        if self.weight_decay == "L1":
            dW -= decay * np.sign(self.W)
        elif self.weight_decay == "L2":
            dW -= decay * self.W

        # can we use BLAS here ?
        # momentum
        # note:  what do we do if lr changes per step ? not ready yet
        dW += self.momentum * self.dW_prev  
        # update
        self.W += lr * dW 

        # storage for next iteration

        # is this is a memory killer 
        self.dW_prev =  dW  
        
        # is this wasteful...can we avoid storing 2X the W mat ?
        # elementwise multiply
        self.W2 = np.multiply(self.W,self.W)

        # update bias terms
        #   csr matrix sum is screwy, returns [[1,self.n_components]] 2-d array  
        #   so I always use np.asarray(X.sum(axis=0)).squeeze()
        #   although (I think) this could be optimized
        self.v_bias += lr * (np.asarray(v_pos.sum(axis=0)).squeeze() -
                             np.asarray(v_neg.sum(axis=0)).squeeze())
        self.h_bias += lr * (np.asarray(h_pos.sum(axis=0)).squeeze() -
                             np.asarray(h_neg.sum(axis=0)).squeeze())

        return 0

    def fit(self, X, y=None):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : EMF_RBM
            The fitted model.
        """
        verbose = self.verbose
        monitor = self.monitor
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        self.random_state = check_random_state(self.random_state)
        
        self.init_weights(X)
        
        n_samples = X.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))

        begin = time.time()
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice])

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end
                
            if monitor:
                print("computing TAP Free Energies")
                fe, [s, u, o] = self._free_energy_TAP(X)
                self.free_energies.append(np.mean(fe))
                self.entropies.append(np.mean(s))
                self.mean_field_energies.append(np.mean(u))
                print("monitor: ", np.mean(fe),  np.mean(s), np.mean(u))
            
        return self
    
    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.
        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self, "W")

        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        return self._mean_hiddens(X)
    
    
