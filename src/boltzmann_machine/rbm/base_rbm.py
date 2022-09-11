from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import math


from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)


import torch.nn.functional as F
from torch.autograd import Function

from ..ebm import EnergyBasedModel

from ...layers import BernoulliLayer



class BaseRBM(EnergyBasedModel):

    def __init__(
        self,
        n_vis: int = 784, 
        v_layer_cls: type = BernoulliLayer, 
        v_layer_params: dict = None,
        n_hid: int = 256, 
        h_layer_cls: type = BernoulliLayer, 
        h_layer_params: dict = None,
        W_init: float = 0.01, 
        vb_init: int = 0., 
        hb_init: float = 0., 
        n_gibbs_steps: int | list[int] = 1,
        sample_v_states: bool = False, 
        sample_h_states: bool = True, 
        p_dropout: float = None,
        sparsity_target: float = 0.1, 
        sparsity_cost: float = 0., 
        sparsity_damping: float = 0.9,
        dbm_first: bool = False, 
        dbm_last: bool = False,
        metrics_config=None, 
        verbose=True, 
        save_after_each_epoch=True,
        display_filters=0, 
        display_hidden_activations=0, 
        v_shape=(28, 28),
        model_path='rbm_model/', 
        strategy: str = "CD",
        *args: Any, 
        **kwargs: Any,
    ):
        super(BaseRBM, self).__init__(*args, **kwargs)
        self.n_vis = n_vis
        self.n_hid = n_hid

        self._weight = nn.Parameter(torch.randn(self.n_hid,n_vis)*1e-2)
        self._v_bias = nn.Parameter(torch.zeros(self.n_vis))
        self._h_bias = nn.Parameter(torch.zeros(self.n_hid))
        self.n_gibbs_steps = [n_gibbs_steps] if (n_gibbs_steps, int) else n_gibbs_steps

        self.reset_parameters()


        # set visible layer params
        v_layer_params = v_layer_params or {}
        v_layer_params.setdefault('n_units', self.n_vis)
        v_layer_params.setdefault('dtype', self.dtype)


        # set hidden layer params
        h_layer_params = h_layer_params or {}
        h_layer_params.setdefault('n_units', self.n_hid)
        h_layer_params.setdefault('dtype', self.dtype)


        self._v_layer = v_layer_cls(**v_layer_params)
        self._h_layer = h_layer_cls(**h_layer_params)

        self.W_init = W_init
        if hasattr(self.W_init, '__iter__'):
            # TODO: check shape of w_init
            self.W_init = np.asarray(self.W_init)

        # Visible biases can be initialized with list of values,
        # because it is often helpful to initialize i-th visible bias
        # with value log(p_i / (1 - p_i)), p_i = fraction of training
        # vectors where i-th unit is on, as proposed in [2]
        self.vb_init = vb_init
        if hasattr(self.vb_init, '__iter__'):
            self.vb_init = np.asarray(self.vb_init)

        self.hb_init = hb_init
        if hasattr(self.hb_init, '__iter__'):
            self.hb_init = np.asarray(self.hb_init)


        # According to [2], the training goes less noisy and slightly faster, if
        # sampling used for states of hidden units driven by the data, and probabilities
        # for ones driven by reconstructions, and if probabilities (means) used for visible units,
        # both driven by data and by reconstructions. It is therefore recommended to set
        # these parameter to False (default).
        self.sample_h_states = sample_h_states
        self.sample_v_states = sample_v_states
        self._p_dropout = p_dropout

        self.sparsity_target = sparsity_target
        self.sparsity_cost = sparsity_cost
        self.sparsity_damping = sparsity_damping

        self.dbm_first = dbm_first
        self.dbm_last = dbm_last


        self._propup_multiplier = 1 + self.dbm_first
        self._propdown_multiplier = 1 + self.dbm_last


        self.verbose = verbose
        self._strategy = strategy
        self.save_after_each_epoch = save_after_each_epoch



        self._v_samples = None

        assert self.n_hid >= display_filters
        self.display_filters = display_filters

        assert self.n_hid >= display_hidden_activations
        self.display_hidden_activations = display_hidden_activations

        self.v_shape = v_shape
        if len(self.v_shape) == 2:
            self.v_shape = (self.v_shape[0], self.v_shape[1], 1)


    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self._weight.size(1))
        nn.init.uniform_(self._weight, -bound, bound)
        if self._v_bias is not None:
            nn.init.uniform_(self._v_bias, -bound, bound)


    def _propup(self, v):
        return torch.matmul(v, self._weight.t())

    def _propdown(self, h):
        return torch.matmul(h, self._weight)

    def _h_given_v(self, v):

        x = self._propup_multiplier * self._propup(v)
        h_bias = self._propup_multiplier * self._h_bias

        h_states = h_means = self._h_layer.activation(x, h_bias)

        if self.sample_h_states:
            h_states = self._h_layer.sample(h_means)


        return h_states, h_means


    def _v_given_h(self, h):

        x = self._propdown_multiplier * self._propdown(h)
        v_bias = self._propdown_multiplier * self._v_bias

        v_states = v_means = self._v_layer.activation(x, v_bias)

        if self.sample_v_states:
            v_states = self._v_layer.sample(v_means)


        return v_states, v_means


    def run_gibbs_step(self, h_states):

        v_states, v_means = self._v_given_h(h_states)
        h_states, h_means = self._h_given_v(v_states)

        return v_states, v_means, h_states, h_means

    
    def run_gibbs_chain(self, h_states):

        v_states = v_means = h_means = None
        
        # TODO: handle variable gibbs step-case
        
        if len(self.n_gibbs_steps) == 1:
            k_iter = range(self.n_gibbs_steps[0])
        else:
            k_iter = range(self.n_gibbs_steps[0])

        for _ in k_iter:
            v_states, v_means, h_states, h_means = self.run_gibbs_step(h_states)
        return v_states, v_means, h_states, h_means

    def get_pseudo_likelihood_cost(self, v):
        """Stochastic approximation to the pseudo-likelihood"""

        N = v.size()[0]   

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = torch.randint(0, self.n_vis, (N,))

        # calculate free energy for the given bit configuration
        fe_v = self._free_energy(v)

        v_flip = v.detach().clone()


        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        v_flip[torch.arange(N), bit_i_idx] = 1 - v_flip[torch.arange(N), bit_i_idx]


        # calculate free energy with bit flipped
        fe_v_flip = self._free_energy(v_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = torch.mean(self.n_vis * F.logsigmoid(fe_v_flip -
                                                            fe_v))

        return cost
    

    def forward(self, input, v_samples = None):

        N = input.size()[0]

        

        if self._p_dropout is not None:
            input = F.dropout(input, p=self._p_dropout, training=self.training)
        
        if self._strategy == "CD":
            v0_states = input

        elif self._strategy == "PCD":
            if v_samples is None:
                if self._v_samples is None:
                    self._v_samples = torch.rand(N, self.n_vis, device=self._weight.device)
                v0_states = self._v_samples[:N]
            else:
                v0_states = v_samples
            
        else:
            raise ValueError("Invalid RBM training strategy...")

        
        # calculate positive phase hidden means
        # input is different from v0_states in PCD
        _, h0_means = self._h_given_v(input)


        # run gibbs chain
        h0_states, _ = self._h_given_v(v0_states)
        v_states, v_means, h_states, h_means = self.run_gibbs_chain(h0_states)



        if self.training:

            dW_positive = torch.matmul(h0_means.t(), input) 
            dW_negative = torch.matmul(h_means.t(), v_states)
            
            self._weight.grad = - (dW_positive - dW_negative) / N 

            self._v_bias.grad = - torch.mean(input - v_states, dim=0)
            self._h_bias.grad = - torch.mean(h0_means - h_means, dim=0)

        if self._strategy == "PCD":
            # update persistent visible states
            
            if v_samples is None:
                self._v_samples[:N] = v_states
        
        return v_states, v_means, h_states, h_means







