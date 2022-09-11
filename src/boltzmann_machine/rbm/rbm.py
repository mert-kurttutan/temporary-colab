import numpy as np
import torch
import torch.nn.functional as F
from .base_rbm import BaseRBM
from ...layers import BernoulliLayer, MultinomialLayer, GaussianLayer


from torch.distributions import multinomial

class BernoulliRBM(BaseRBM):
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, *args, **kwargs):
        super(BernoulliRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                           h_layer_cls=BernoulliLayer,
                                          *args, **kwargs)

    def _free_energy(self, v):
        T1 = -torch.matmul(v, self._v_bias)
        T2 = -torch.sum(F.softplus(self._propup(v) + self._h_bias), axis=1)
        fe = torch.mean(T1 + T2, axis=0)
        return fe


class MultinomialRBM(BaseRBM):
    """RBM with Bernoulli visible and single Multinomial hidden unit
    (= multiple softmax units with tied weights).

    Parameters
    ----------
    n_hidden : int
        Number of possible states of a multinomial unit.
    n_samples : int
        Number of softmax units with shared weights
        (= number of samples from one softmax unit).

    References
    ----------
    [1] R. Salakhutdinov, A. Mnih, and G. Hinton. Restricted boltzmann
        machines for collaborative filtering, 2007.
    """
    def __init__(self, n_samples=100,
                 model_path='m_rbm_model/', *args, **kwargs):
        self.n_samples = n_samples
        super(MultinomialRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                             h_layer_cls=MultinomialLayer,
                                             h_layer_params=dict(n_samples=self.n_samples),
                                             model_path=model_path, *args, **kwargs)

    def _free_energy(self, v):
        K = float(self.n_hid)
        M = float(self.n_samples)

        T1 = -torch.matmul(v, self._v_bias)
        T2 = -torch.matmul(v, self._weight)
        h_hat = multinomial.Multinomial(num_samples=M, logits=torch.ones([K])).sample()
        T3 = torch.matmul(T2, h_hat)
        fe = torch.mean(T1 + T3, axis=0)
        fe += -torch.lgamma(M + K) + torch.lgamma(M + 1) + torch.lgamma(K)
        return fe

    def transform(self, *args, **kwargs):
        H = super(MultinomialRBM, self).transform(*args, **kwargs)
        H /= float(self.n_samples)
        return H


class GaussianRBM(BaseRBM):
    """RBM with Gaussian visible and Bernoulli hidden units.

    This implementation does not learn variances, but instead uses
    fixed, predetermined values. Input data should be pre-processed
    to have zero mean (or, equivalently, initialize visible biases
    to the negative mean of data). It can also be normalized to have
    unit variance. In the latter case use `sigma` equal to 1., as
    suggested in [1].

    Parameters
    ----------
    sigma : float, or iterable of such
        Standard deviations of visible units.

    References
    ----------
    [1] Hinton, G. "A Practical Guide to Training Restricted Boltzmann
        Machines" UTML TR 2010-003
    """
    def __init__(self, learning_rate=1e-3, sigma=1.,
                 *args, **kwargs):
        self.sigma = sigma
        super(GaussianRBM, self).__init__(v_layer_cls=GaussianLayer,
                                          v_layer_params=dict(sigma=self.sigma),
                                          h_layer_cls=BernoulliLayer,
                                          learning_rate=learning_rate,
                                           *args, **kwargs)
        if hasattr(self.sigma, '__iter__'):
            self._sigma = self.sigma = np.asarray(self.sigma)
        else:
            self._sigma = np.repeat(self.sigma, self.n_vis)



    def _free_energy(self, v):
        T1 = torch.divide(torch.reshape(self._v_bias, [1, self.n_vis]), self._sigma)
        T2 = torch.square(torch.subtract(v, T1))
        T3 = 0.5 * torch.sum(T2, axis=1)
        T4 = -torch.sum(F.softplus(self._propup(v) + self._h_bias), axis=1)
        fe = torch.mean(T3 + T4, axis=0)
        return fe


