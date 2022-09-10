import numpy as np
import torch 



class BaseMixin(object):
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise AttributeError(f'Invalid parameters: {args}, {kwargs}')
        super(BaseMixin, self).__init__()


class DtypeMixin(BaseMixin):
    def __init__(self, dtype='float32', *args, **kwargs):
        super(DtypeMixin, self).__init__(*args, **kwargs)
        self.dtype = dtype

    @property
    def _torch_dtype(self):
        return getattr(torch, self.dtype)

    @property
    def _np_dtype(self):
        return getattr(np, self.dtype)


class SeedMixin(BaseMixin):
    def __init__(self, random_seed=None, *args, **kwargs):
        super(SeedMixin, self).__init__(*args, **kwargs)
        self.random_seed = random_seed


