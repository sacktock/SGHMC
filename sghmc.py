from numpy import corrcoef
import torch

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.ops.integrator import potential_grad

from util import initialize_model, observed_information
from param_tensor_corresponder import ParamTensorCorresponder
from dual_averaging_step_size import DualAveragingStepSize
from collections import OrderedDict

class SGHMC(MCMCKernel):
    """Stochastic Gradient Hamiltonian Monte Carlo kernel.
    
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
        
    with_friction : bool, default=True
        Use friction term when updating momentum

    resample_every_n : int, default 50
        When to resmaple to momentum (deafult is to resample momentum every 50 samples)

    friction_term : dict or None, default=None
        The friction term to use for the Langevin dynamics. This should be a 
        dictionary of square tensors, one for each of the model parameters,
        with the size of each matrix determined by the dimension of each
        parameter. The default is to use identity matrices for each
        parameter.

    obs_info_noise : bool, default=True
        Use the observed information to estimate the noise model

    compute_obs_info : string, default="every_sample", valid=["start", "every_sample", "every_step"]
        When to compute the observed information matrix to estimate B hat,
        - "start" once at the begining using the inital parameters
        - "every_sample" once at the start of every sampling procedure
        - "every_step" at every intehration step or when the parameters change

    do_mh_correction : bool, default False
        Compute the mh correction term using the whole dataset
        
    do_step_size_adaptation : bool, default True
        Do step size adaptation during warm up phase

    target_accept : float, deafult 0.8
        Target acceptance ratio for tuning the step size

    Limitations
    -----------

    - `friction_term` must be constant, and can't vary according to the values
    of the parameters.
    - `friction_term` yields a block matrix: friction is applied independently
    to each parameter. One parameter's values can't affect the friction on
    another.
    - The observed information is computed only once per sample, and is not
    updated while simulating the dynamics.
    """

    def __init__(self, 
                 model, 
                 subsample_positions=[0], 
                 batch_size=5, 
                 step_size=0.1, 
                 num_steps=10, 
                 resample_every_n=50, 
                 with_friction=True, 
                 friction_term=None, 
                 friction_constant=1.0, 
                 obs_info_noise=True, 
                 compute_obs_info='every_sample', 
                 do_mh_correction=False, 
                 do_step_size_adaptation=False, 
                 target_accept=0.8):
      
        self.model = model
        self.subsample_positions = subsample_positions
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_steps = num_steps
        self.resample_every_n = resample_every_n
        self.with_friction = with_friction
        self.friction_term = friction_term
        self.C = friction_constant if not friction_term else None
        self.obs_info_noise = obs_info_noise
        self.compute_obs_info = compute_obs_info
        self.do_mh_correction = do_mh_correction
        self.do_step_size_adaptation = do_step_size_adaptation
        self.target_accept = target_accept
        self._initial_params = None
        self.corresponder = ParamTensorCorresponder()
        
        
    def setup(self, warmup_steps, *model_args, **model_kwargs):
        self._warmup_steps = warmup_steps

        # Save positional and keyword arguments
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        
        # Compute the data size and check if it is a pytorch tensor
        try:
            self.data_size = self.model_args[self.subsample_positions[0]].size(0)
        except AttributeError:
            raise RuntimeError("Positional argument {} is not a pytorch tensor with size attribute".format(self.subsample_positions[0]))
            
        # Check all the data is the same length, otherwise we can't meaningfully subsample
        for pos in self.subsample_positions[1:]:
            try:
                assert self.data_size == self.model_args[pos].size(0)
            except AttributeError:
                raise RuntimeError("Positional argument {} is not a pytorch tensor with size attribute".format(pos))
            except AssertionError:
                raise RuntimeError("Can't subsample arguments with different lengths")

        # Check compute_obs_info is valid
        try:
            assert self.compute_obs_info in ["start", "every_sample", "every_step"]
        except AssertionError:
            raise RuntimeError("Invalid input for compute_obs_info "+str(self.compute_obs_info))
                
        # Compute the initial parameter and potential function from the model
        # Use entire dataset to find initial parameters using pyros in-built search    
        if self.compute_obs_info == "start":
            # We need the nll_fn if we want to compute obs info right now
            initial_params, potential_fn, nll_fn, transforms, _ = initialize_model(
                self.model,
                self.model_args,
                self.model_kwargs,
                initial_params=self._initial_params,
                return_nll_fn=True
            )
        else:
            initial_params, potential_fn, transforms, _ = initialize_model(
                self.model,
                self.model_args,
                self.model_kwargs,
                initial_params = self._initial_params
            )

        # Cache variables
        self._initial_params = initial_params
        self.full_potential_fn = potential_fn
        self.transforms = transforms

        # Set up the corresponder between parameter dicts and tensors
        self.corresponder.configure(initial_params)

        if self.compute_obs_info == "start":
            # Compute obs_info once using initial parameters
            self.obs_info = self.compute_observed_information(initial_params, nll_fn)
        else:
            # Set up the obs_info variable
            self.obs_info = None

        self._obs_info_arr = []

        # Compute the friction term as a parameter dict and a block matrix
        if self.with_friction:
            if self.friction_term is None:
                self.friction_term = {}
                for name, size in self.corresponder.site_sizes.items():
                    self.friction_term[name] = torch.eye(size) * self.C
            self.friction_term_tensor = self.corresponder.to_block_matrix(
                self.friction_term
            )
        
        # Set up the automatic step size adapter
        if self.do_step_size_adaptation:
            self.step_size_adapter = DualAveragingStepSize(self.step_size, target_accept=self.target_accept)

        # Set the step counter to 0
        self._step_count = 0

    # Computes orig + step * mom elementwise, where orig and mom are
    # dictionaries with the same keys
    def _step_position(self, orig, mom):   
        return {site:(orig[site] + self.step_size * mom[site]) for site in orig}
        
    # Computes orig + step * grad elementwise, where orig and grad are
    # dictionaries with the same keys
    # If with_friction adds additional update terms elementwise
    def _step_momentum(self, orig, grad):   
        if self.with_friction:
            f = self._sample_friction(f"f_{self._step_count}")
            return {site:(orig[site] - self.step_size * grad[site] - self.step_size * self.friction_term[site] * orig[site] + f[site]) for site in orig}
        else:
            return {site:(orig[site] - self.step_size * grad[site]) for site in orig}

    # Sample the momentum variables from a standard normal
    def sample_momentum(self, sample_name):
        loc = torch.zeros(self.corresponder.total_size)
        scale = torch.ones(self.corresponder.total_size)
        sample = pyro.sample(sample_name, dist.Normal(loc, scale))
        return self.corresponder.to_params(sample)

    # Get the potential and negative log likelihood functions for a minibatch
    def get_potential_nll_functions(self, subsample=False):
        if subsample:
            model_args_lst = list(self.model_args).copy()
            
            # Sample random indices
            perm = torch.randperm(self.data_size)
            idx = perm[:self.batch_size]
            
            # Sample a random mini batch for each subsampled argument
            for pos in self.subsample_positions:
                model_args_lst[pos] = model_args_lst[pos][idx]
  
            model_args = tuple(model_args_lst)
            batch_size = self.batch_size
        else:
            batch_size = self.data_size
            model_args = self.model_args
            
        _, potential_fn, nll_fn, _, _ = initialize_model(
            self.model,
            model_args,
            self.model_kwargs,
            initial_params=self._initial_params,
            scale_likelihood=self.data_size/batch_size,
            return_nll_fn=True
        )
        return potential_fn, nll_fn

    def compute_observed_information(self, q, nll_fn):
        """Compute the observed information at position `q`."""

        # The position as a flat tensor
        q_tensor = self.corresponder.to_tensor(q)

        # The negative log likehood, modified to accept tensors
        def nll_fn_tensor(q_tensor):
            return nll_fn(self.corresponder.to_params(q_tensor))

        return observed_information(nll_fn_tensor, q_tensor)

    # Sample the momentum variables from a standard normal
    def _sample_friction(self, sample_name):

        # Only use the noise term if we have computed it from the observed
        # information
        if self.obs_info_noise:
            noise_term = 0.5 * self.step_size * self.obs_info
        else:
            noise_term = 0
        
        # Determine the scale and covariace of the friction
        loc = torch.zeros(self.corresponder.total_size)
        cov = (2 * (self.friction_term_tensor - noise_term) * self.step_size)

        # Sample the friction
        sample = pyro.sample(sample_name, dist.MultivariateNormal(loc, cov))

        # Return it as a parameter dictionary
        return self.corresponder.to_params(sample)
        
    # Update the position one step

    def update_position(self, p, q):
        return self._step_position(q, p)

    # Update the momentum one step
    def update_momentum(self, p, q, potential_fn):
        # Compute the partial derivative with respect to the position
        grad_q, _ = potential_grad(potential_fn, q)
        return self._step_momentum(p, grad_q)

    # Compute the kinetic energy, given the momentum
    def kinetic_energy(self, p):
        energy = torch.zeros(1)
        for site, value in p.items():
            energy += torch.dot(value, value)
        return 0.5 * energy

    def sample(self, params):
        # Increment the step counter
        self._step_count += 1

        # Compute the new potential fn
        potential_fn, nll_fn = self.get_potential_nll_functions(subsample=True)

        # Position variable
        q = q_current = params

        # Resample the momentum
        if not ((self._step_count - 1) % self.resample_every_n):
            p = p_current = self.sample_momentum(f"p_{self._step_count}")
        else:
            p = p_current = self._p_current

        # Compute obs info at the start of a new sample
        if self.obs_info_noise and self.compute_obs_info in ["every_step", "every_sample"]:
            self.obs_info = self.compute_observed_information(q, nll_fn)
            self._obs_info_arr += [self.obs_info]

        # Full-step position and momentum alternately
        for i in range(self.num_steps):
            # Compute obs info evey leapfrog step
            if self.obs_info_noise and self.compute_obs_info == "every_step" and i > 0:
                self.obs_info = self.compute_observed_information(q, nll_fn)
                self._obs_info_arr += [self.obs_info]
            q = self.update_position(p, q)
            p = self.update_momentum(p, q, potential_fn)

        # Cache current momentum 
        self._p_current = p
        
        ## Metropolis-Hastings correction
        if self.do_mh_correction:
            # Compute the acceptance probability
            accept_prob = self._compute_accept_prob(self.full_potential_fn, q_current, p_current, q, p)

            # Draw a uniform random sample
            rand = pyro.sample(f"rand_{self._step_count}", dist.Uniform(0,1))

            # Update the step size with the step size adapter using the true acceptance prob
            if self._step_count <= self._warmup_steps and self.do_step_size_adaptation:
                self._update_step_size(accept_prob)

            # Accept the new point with probability `accept_prob`
            if rand < accept_prob:
                return q
            else:
                return q_current
        else:
            # Update the step size with the step size adapter using a noisy acceptance prob
            if self._step_count <= self._warmup_steps and self.do_step_size_adaptation:
                accept_prob = self._compute_accept_prob(potential_fn, q_current, p_current, q, p)
                self._update_step_size(accept_prob)

            return q

    def get_obs_info_arr(self):
        return np.array(self._obs_info_arr)

    def _compute_accept_prob(self, potential_fn, q_current, p_current, q, p):
        # squeeze params to 1d tensors
        q_current = self.corresponder.squeeze_params_to_1d(q_current)
        p_current = self.corresponder.squeeze_params_to_1d(p_current)
        q = self.corresponder.squeeze_params_to_1d(q)
        p = self.corresponder.squeeze_params_to_1d(p)

        energy_current = (potential_fn(q_current) 
                              + self.kinetic_energy(p_current))
        energy_proposal = potential_fn(q) + self.kinetic_energy(p)

        # Compute the acceptance probability
        energy_delta = energy_current - energy_proposal
        return energy_delta.exp().clamp(max=1.0).item()

    # Method to update the step size if we have the acceptance probability handy
    def _update_step_size(self, accept_prob):
        if self._step_count < self._warmup_steps:
            step_size, _ = self.step_size_adapter.update(accept_prob)
        elif self._step_count == self._warmup_steps:
            _, step_size = self.step_size_adapter.update(accept_prob)
        
        self.step_size = step_size
        
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