import sys
sys.path.append("..")

from numpy import corrcoef
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.ops.integrator import potential_grad
from kernel.utils.main import initialize_model, observed_information
from kernel.utils.param_tensor_corresponder import ParamTensorCorresponder
from kernel.utils.dual_averaging_step_size import DualAveragingStepSize
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

    resample_every_n : int, default 50
        When to resmaple to momentum (deafult is to resample momentum every 50 samples)

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
                 learning_rate=0.1, 
                 momentum_decay=0.01,
                 num_steps=10, 
                 resample_every_n=50, 
                 obs_info_noise=False, 
                 compute_obs_info=None
                 ):
      
        self.model = model
        self.subsample_positions = subsample_positions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.num_steps = num_steps
        self.resample_every_n = resample_every_n
        self.obs_info_noise = obs_info_noise
        self.compute_obs_info = compute_obs_info
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
            assert self.compute_obs_info in ["start", "every_sample", "every_step", None]
        except AssertionError:
            raise RuntimeError("Invalid input for compute_obs_info "+str(self.compute_obs_info))
                
        # Compute the initial parameter and potential function from the model
        # Use entire dataset to find initial parameters using pyros in-built search    
        if self.compute_obs_info == "start" and self.obs_info_noise:
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


        # Set up the corresponder between parameter dicts and tensors
        self.corresponder.configure(initial_params)

        #initial_params = self.corresponder.wrap(initial_params)

        # Cache variables
        self._initial_params = initial_params
        self.full_potential_fn = potential_fn
        self.transforms = transforms

        if self.compute_obs_info == "start" and self.obs_info_noise:
            # Compute obs_info once using initial parameters
            self.obs_info = self.compute_observed_information(initial_params, nll_fn)
        else:
            # Set up the obs_info variable
            self.obs_info = None

        self._obs_info_arr = []

        # Set the step counter to 0
        self._step_count = 0

    # Computes orig + step * mom elementwise, where orig and mom are
    # dictionaries with the same keys
    def _step_position(self, orig, mom): 
        return {site:(orig[site] + mom[site].view(orig[site].shape)) for site in orig}
        
    # Computes orig + step * grad elementwise, where orig and grad are
    # dictionaries with the same keys
    # If with_friction adds additional update terms elementwise
    def _step_momentum(self, orig, grad):   
        fric = self._sample_friction(f"f_{self._step_count}")
        return {site:(
                      (1 - self.momentum_decay) * orig[site].view(grad[site].shape)
                       - self.learning_rate * grad[site] + 
                       fric[site].view(grad[site].shape)) 
                for site in orig}

    # Sample the momentum variables from a standard normal
    def sample_momentum(self, sample_name):
        loc = torch.zeros(self.corresponder.total_size)
        scale = torch.ones(self.corresponder.total_size) * np.sqrt(self.learning_rate)
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
        
        nll_fn = None

        if self.obs_info_noise:
            _, potential_fn, nll_fn, _, _ = initialize_model(
                self.model,
                model_args,
                self.model_kwargs,
                initial_params=self._initial_params,
                scale_likelihood=self.data_size/batch_size,
                return_nll_fn=True
            )
        else:
            _, potential_fn, _, _ = initialize_model(
                self.model,
                model_args,
                self.model_kwargs,
                initial_params=self._initial_params,
                scale_likelihood=self.data_size/batch_size,
                return_nll_fn=False
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
            noise_term = torch.diagonal(0.5 * self.learning_rate * self.obs_info)
        else:
            noise_term = torch.zeros(self.corresponder.total_size)

        loc = torch.zeros(self.corresponder.total_size)
        scale = torch.ones(self.corresponder.total_size) * (self.momentum_decay - noise_term) * 2 * self.learning_rate
        
        # Sample the friction
        sample = pyro.sample(sample_name, dist.Normal(loc, scale))

        # Return it as a parameter dictionary
        return self.corresponder.to_params(sample)
        
    # Update the position one step

    def update_position(self, theta, v):
        return self._step_position(theta, v)

    # Update the momentum one step
    def update_momentum(self, v, theta, potential_fn):
        # Compute the partial derivative with respect to the position
        grad, _ = potential_grad(potential_fn, theta)
        return self._step_momentum(v, grad)

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
        theta = params

        # Resample the momentum
        if not ((self._step_count - 1) % self.resample_every_n):
            v = self.sample_momentum(f"v_{self._step_count}")
        else:
            v = self._momentum

        # Compute obs info at the start of a new sample
        if self.obs_info_noise and self.compute_obs_info =="every_sample":
            self.obs_info = self.compute_observed_information(theta, nll_fn)
            self._obs_info_arr += [self.obs_info]

        # Full-step position and momentum alternately
        for i in range(self.num_steps):
            # Compute obs info evey leapfrog step
            if self.obs_info_noise and self.compute_obs_info == "every_step":
                self.obs_info = self.compute_observed_information(theta, nll_fn)
                self._obs_info_arr += [self.obs_info]

            theta = self.update_position(theta, v)
            v = self.update_momentum(v, theta, potential_fn)

        # Cache current momentum 
        self._momentum = v

        return theta

    def get_obs_info_arr(self):
        return np.array(self._obs_info_arr)
        
    def logging(self):
        return OrderedDict(
            [
                ("lr", "{:.2e}".format(self.learning_rate)),
            ]
        )

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params