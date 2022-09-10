from ..base import DtypeMixin
import torch.nn as nn
import torch.nn.functional as F



class EnergyBasedModel(DtypeMixin, nn.Module):
    """A generic Energy-based model with hidden variables."""
    def __init__(self, *args, **kwargs):
        super(EnergyBasedModel, self).__init__(*args, **kwargs)

    def _free_energy(self, v):
        """
        Compute (average) free energy of a visible vectors `v`.

        Parameters
        ----------
        v : (batch_size, n_visible) tf.Tensor
        """
        raise NotImplementedError('`free_energy` is not implemented')


    def free_energy(self,v):
        vbias_term = v.mv(self._v_bias)
        wx_b = F.linear(v, self._weight, self._h_bias)

        # exponentiate to get probabilities
        hidden_term = wx_b.exp()
        
        # add one to account for h_i=0
        # log and sum to sum over hidden states
        # negative log-likelihood
        NLL_ = - hidden_term.add(1).log().sum(1) - vbias_term

        # mean over batch_dim
        return NLL_.mean()
