import torch

import pyro
import pyro.distributions as dist

class ParamTensorCorresponder():
    """A class for converting between dicts of parametes and tenors.
    
    In Pyro, we work with dictionaries of named parameters, whose values are
    PyTorch tensors. Sometimes we need to convert these to flat 1D tensors by
    concatenation, and back again. This class allows this process to be done 
    consistently.
    """

    def __init__(self):
        pass

    def configure(self, initial_params):
        """Use the initial parameters to set up the corresponder."""
        self._site_names = sorted(initial_params.keys())
        self._site_shapes = {}
        self._site_sizes = {}
        for name in self._site_names:
            self._site_shapes[name] = initial_params[name].shape
            self._site_sizes[name] = initial_params[name].numel()
        self._total_size = sum(self._site_sizes.values())

    def to_tensor(self, params):
        """Convert a parameter dict into a flat tensor."""
        cat_list = []
        for name in self._site_names:
            cat_list.append(params[name].detach().reshape(-1))
        return torch.cat(cat_list)

    def to_params(self, tensor):
        """Convert a tensor into a parameter dict."""
        params = {}
        tensor_pos = 0
        for name in self._site_names:
            shape = self._site_shapes[name]
            size = self._site_sizes[name]
            flattened = tensor[tensor_pos:tensor_pos + size]
            params[name] = flattened.detach().reshape(shape)
            tensor_pos += size
        return params

    def to_block_matrix(self, params):
        """Convert a parameter dict of matrices to a block matrix tensor."""
        block_list = []
        for name in self._site_names:
            block_list.append(params[name].detach())
        return torch.block_diag(*block_list)

    def wrap(self, params):
        """Remove all dimensions with size 1 from dictionary of parameters apart from the first dimension"""
        for name in self._site_names:
            params[name] = params[name].squeeze().unsqueeze(0)

        return params

    def zeros_params(self):
        """Return a params dict full of zeros."""
        return self.to_params(torch.zeros(self.total_size))

    def ones_params(self):
        """Return a params dict full of ones."""
        return self.to_params(torch.ones(self.total_size))

    @property
    def site_shapes(self):
        return self._site_shapes

    @property
    def site_sizes(self):
        return self._site_sizes

    @property
    def total_size(self):
        return self._total_size