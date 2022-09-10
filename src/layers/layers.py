import numpy as np
import torch
import torch.nn as nn
import math

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

    def _sample(self, p):
        """Sample states of the units by combining output from 2 previous functions."""
        raise NotImplementedError('`sample` is not implemented')

    def sample(self, p):
        return self._sample(p).type(self._torch_dtype)

class BernoulliLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(BernoulliLayer, self).__init__(*args, **kwargs)


    def activation(self, x, b):
        return torch.sigmoid(x + b)
    

    def _sample(self, means):
        return torch.bernoulli(means)
    


class MultinomialLayer(BaseLayer):
    def __init__(self, n_samples=100, *args, **kwargs):
        super(MultinomialLayer, self).__init__(*args, **kwargs)
        self.n_samples = n_samples

    def activation(self, x, b):
        return self.n_samples * torch.nn.softmax(x + b)

    def _sample(self, means):
        probs = (means / means.sum()).type(torch.float)
        return torch.multinomial(input=probs, num_samples=self.n_samples, replacement=True)


class GaussianLayer(BaseLayer):
    def __init__(self, sigma, *args, **kwargs):
        super(GaussianLayer, self).__init__(*args, **kwargs)
        self.sigma = torch.tensor([sigma])

    def activation(self, x, b):
        t = x * self.sigma + b
        return t

    def _sample(self, means):
        return torch.normal(mean=means, std=self.sigma.type(self._torch_dtype))
