from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import math
from torch.distributions import bernoulli, multinomial, normal

from ..base import DtypeMixin


class BaseLayer(DtypeMixin):
    """Class encapsulating one layer of stochastic units."""
    def __init__(self, n_units, *args, **kwargs):
        super(BaseLayer, self).__init__(*args, **kwargs)
        self.n_units = n_units

    @staticmethod
    def activation(x, b):
        """Compute activation of states according to their distribution.

        Parameters
        ----------
        x : (n_units,) tf.Tensor
            Total input received (excluding bias).
        b : (n_units,) tf.Tensor
            Bias.
        """
        raise NotImplementedError('`activation` is not implemented')

    def _sample(self, p) -> bernoulli.Bernoulli | multinomial.Multinomial | normal.Normal:
        """Sample states of the units by combining output from 2 previous functions."""
        raise NotImplementedError('`sample` is not implemented')

    def sample(self, p):
        return self._sample(p).sample().type(self._torch_dtype)

class BernoulliLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(BernoulliLayer, self).__init__(*args, **kwargs)


    def activation(self, x, b):
        return torch.sigmoid(x + b)
    

    def _sample(self, means):
        return bernoulli.Bernoulli(probs=means)
    


class MultinomialLayer(BaseLayer):
    def __init__(self, n_samples=100, *args, **kwargs):
        super(MultinomialLayer, self).__init__(*args, **kwargs)
        self.n_samples = n_samples

    def activation(self, x, b):
        return self.n_samples * torch.nn.softmax(x + b)

    def _sample(self, means):
        probs = (means / means.sum()).type(torch.float)
        return multinomial.Multinomial(probs=probs, total_count=self.n_samples)


class GaussianLayer(BaseLayer):
    def __init__(self, sigma, *args, **kwargs):
        super(GaussianLayer, self).__init__(*args, **kwargs)
        self.sigma = torch.tensor([sigma])

    def activation(self, x, b):
        t = x * self.sigma + b
        return t

    def _sample(self, means):
        return normal.Normal(loc=means, scale=self.sigma.type(self._torch_dtype))
