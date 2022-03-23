
from sghmc import SGHMC
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

    step_size : int, default 0.1
        The size of a single step taken while simulating Hamiltonian dynamics

    num_steps : int, default 10
        The number of steps to simulate Hamiltonian dynamics

    obs_info_noise : bool, default=True
        Use the observed information to estimate the noise model

    compute_obs_info : string, default="every_sample", valid=["start", "every_sample", "every_step"]
        When to compute the observed information matrix to estimate B hat,
        - "start" once at the begining using the inital parameters
        - "every_sample" once at the start of every sampling procedure
        - "every_step" at every intehration step or when the parameters change
    """

    def __init__(self, 
                 model, 
                 subsample_positions=[0], 
                 batch_size=5, 
                 step_size=0.1, 
                 num_steps=10, 
                 obs_info_noise=True, 
                 compute_obs_info='every_sample'):

        super().__init__(model, 
                         subsample_positions=subsample_positions, 
                         batch_size=batch_size, 
                         step_size=step_size, 
                         num_steps=num_steps, 
                         with_friction=False,
                         obs_info_noise=obs_info_noise, 
                         compute_obs_info=compute_obs_info)

    def setup(self, warmup_steps, *model_args, **model_kwargs):
        super().setup(warmup_steps, *model_args, **model_kwargs)

    # Computes orig + step *grad + noise elementwise, where orig, grad and noise are
    # dictionaries with the same keys
    def _step_position(self, orig, grad, noise):   
        return {site:(orig[site] - self.step_size * grad[site] + noise[site]) for site in orig}

    def _sample_noise(self, sample_name):
        if self.obs_info_noise:
            noise_term = 0.5 * self.step_size * self.obs_info
        else:
            noise_term = self.step_size

        loc = torch.zeros(self.corresponder.total_size)
        cov = torch.eye(self.corresponder.total_size) * noise_term

        sample = pyro.sample(sample_name, dist.MultivariateNormal(loc, cov))

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
        if self.obs_info_noise and self.compute_obs_info in ["every_step", "every_sample"]:
            self.obs_info = self.compute_observed_information(x, nll_fn)
            self._obs_info_arr += [self.obs_info]

        for i in range(self.num_steps):
            # Compute obs info evey step
            if self.obs_info_noise and self.compute_obs_info == "every_step" and i > 0:
                self.obs_info = self.compute_observed_information(x, nll_fn)
                self._obs_info_arr += [self.obs_info]
            # Step position variable using gradient plus Langevin noise
            x = self.update_position(x, potential_fn)

        return x

    def logging(self):
        return OrderedDict(
            [
                ("step size", "{:.2e}".format(self.step_size))
            ]
        )

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params
    
