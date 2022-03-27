import sys
sys.path.append("..")

from kernel.sghmc import SGHMC
from pyro.ops.integrator import potential_grad
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from collections import OrderedDict

class SGLD(SGHMC):
    """Stochastic Gradient with Langevin Dynamics
    
    Parameters
    ----------
    subsample_positions : list, default [0] / 1st positional argument only
        Specifies which positional arguments of the model to subsample during runtime
        
    batch_size : int, default=5
        The size of the minibatches to use

    learning_rate : int, default 0.1
        The size of a single step taken during sampling

    num_steps : int, default 10
        The number of steps to simulate Hamiltonian dynamics

    obs_info_noise : bool, default=False
        Use the observed information to estimate the noise model

    compute_obs_info : string, default=None, valid=["start", "every_sample", "every_step", None]
        When to compute the observed information matrix to estimate B hat,
        - "start" once at the begining using the inital parameters
        - "every_sample" once at the start of every sampling procedure
        - "every_step" at every intehration step or when the parameters change
    """

    def __init__(self, 
                 model, 
                 subsample_positions=[0], 
                 batch_size=5, 
                 learning_rate=0.1, 
                 noise_rate=0.2,
                 num_steps=10, 
                 obs_info_noise=False, 
                 compute_obs_info=None):
        self.noise_rate = noise_rate
        super().__init__(model, 
                         subsample_positions=subsample_positions, 
                         batch_size=batch_size, 
                         learning_rate=learning_rate, 
                         num_steps=num_steps, 
                         obs_info_noise=obs_info_noise, 
                         compute_obs_info=compute_obs_info)

    def setup(self, warmup_steps, *model_args, **model_kwargs):
        super().setup(warmup_steps, *model_args, **model_kwargs)

    # Computes orig + step *grad + noise elementwise, where orig, grad and noise are
    # dictionaries with the same keys
    def _step_position(self, orig, grad, noise):   
        return {site:(orig[site] - self.learning_rate * grad[site] + noise[site].view(orig[site].shape)) for site in orig}

    def _sample_noise(self, sample_name):
        if self.obs_info_noise:
            noise_term = 0.5 * self.noise_rate * self.obs_info
            loc = torch.zeros(self.corresponder.total_size)
            cov = torch.eye(self.corresponder.total_size) * noise_term
            sample = pyro.sample(sample_name, dist.MultivariateNormal(loc, cov))
        else:
            noise_term = self.noise_rate
            loc = torch.zeros(self.corresponder.total_size)
            scale = torch.ones(self.corresponder.total_size)  * noise_term
            sample = pyro.sample(sample_name, dist.Normal(loc, scale))

        return self.corresponder.to_params(sample)

    def update_position(self, x, potential_fn):
        grad_x, _  = potential_grad(potential_fn, x)
        noise = self._sample_noise(f"noise_{self._step_count}")
        return self._step_position(x, grad_x, noise)

    def sample(self, params):
        # Increment the step counter
        self._step_count += 1

        # Compute the new potential fn
        potential_fn, nll_fn = self.get_potential_nll_functions(subsample=True)

        # Position variable
        x = params

        # Compute obs info at the start of a new sample
        if self.obs_info_noise and self.compute_obs_info == "every_sample":
            self.obs_info = self.compute_observed_information(x, nll_fn)

        for i in range(self.num_steps):
            # Compute obs info evey step
            if self.obs_info_noise and self.compute_obs_info == "every_step":
                self.obs_info = self.compute_observed_information(x, nll_fn)
            # Step position variable using gradient plus Langevin dynamics
            x = self.update_position(x, potential_fn)

        # Cache params
        self._initial_params = x

        return x

    def logging(self):
        return OrderedDict(
            [
                ("lr", "{:.2e}".format(self.learning_rate))
            ]
        )

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params
    
