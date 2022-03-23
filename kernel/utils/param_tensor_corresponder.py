import torch

import pyro
import pyro.distributions as dist

class ParamTensorCorresponder():
    """A class for converting between dicts of parametes and tenors."""

    def __init__(self):
        pass

    def configure(self, initial_params):
        """Use the inital parameters to set up the corresponder."""
        self._site_names = sorted(initial_params.keys())
        self._site_sizes = {}
        for name in self._site_names:
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
            size = self._site_sizes[name]
            params[name] = tensor[tensor_pos:tensor_pos+size]
            tensor_pos += size
        return params

    def to_block_matrix(self, params):
        """Convert a parameter dict of matrices to a block matrix tensor."""
        block_list = []
        for name in self._site_names:
            block_list.append(params[name].detach())
        return torch.block_diag(*block_list)

    def squeeze_params_to_1d(self, params):
        """removes all dimensions with size 1 from dictionary of parameters apart from the first dimension"""
        for name in self._site_names:
            params[name] = params[name].squeeze().unsqueeze(0)

        return params

    @property
    def site_sizes(self):
        return self._site_sizes

    @property
    def total_size(self):
        return self._total_size