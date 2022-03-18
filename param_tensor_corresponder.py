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

    def params_to_tensor(self, params):
        """Convert a parameter dict into a flat tensor."""
        cat_list = []
        for name in self._site_names:
            cat_list.append(params[name].detach().reshape(-1))
        return torch.cat(cat_list)

    def tensor_to_params(self, tensor):
        """Convert a tensor into a parameter dict."""
        params = {}
        tensor_pos = 0
        for name in self._site_names:
            size = self._site_sizes[name]
            params[name] = tensor[tensor_pos:tensor_pos+size]
            tensor_pos += size
        return params

    def normal_sample(self, loc, scale, sample_name):
        """Sample from a normal distribution, returning a parameter dict.
        
        Parameters
        ----------
        loc : tensor (1 dimensional)
            The location tensor
            
        scale : tensor (2 dimensional)
            The covariance matrix

        sample_name : str
            The name for the sample, for Pyro
            
        Return values
        -------------
        sample : dict
            Dictionary of samples for each site name, drawn from the normal
            distribution
        """
        sample_tensor = pyro.sample(sample_name, dist.Normal(loc, scale))
        return self.tensor_to_params(sample_tensor)


    @property
    def site_sizes(self):
        return self._site_sizes

    @property
    def total_size(self):
        return self._total_size