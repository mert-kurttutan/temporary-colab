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

