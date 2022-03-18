import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from util import initialize_model
from pyro.ops.integrator import potential_grad
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

    step_size : int, default=1
        The size of a single step taken while simulating Hamiltonian dynamics

    num_steps : int, default 10
        The number of steps to simulate Hamiltonian dynamics
        
    with_friction : bool, default False
        Use friction term when updating momentum

    do_mh_correction : bool, default False
        Compute the mh correction term using the whole dataset

    do_step_size_adaptation : bool, default True
        Do step size adaptation during warm up phase
    """

    def __init__(self, model, subsample_positions=[0], batch_size=5, step_size=1, num_steps=10,
                 with_friction=True, do_mh_correction=False, do_step_size_adaptation=True):
        self.model = model
        self.subsample_positions = subsample_positions
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_steps = num_steps
        self.do_mh_correction = do_mh_correction
        self.with_friction = with_friction
        self.do_step_size_adaptation = do_step_size_adaptation
        self._initial_params = None
        self.C = 1
        self.B_hat = 0
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
                
        # Compute the initial parameter and potential function from the model
        # Use entire dataset to find initial parameters using pyros in-built search    
        initial_params, potential_fn, transforms, _ = initialize_model(
            self.model,
            self.model_args,
            self.model_kwargs,
            initial_params = self._initial_params
        )
        self._initial_params = initial_params
        self.full_potential_fn = potential_fn
        self.transforms = transforms

        # Set up the corresponder between parameter dicts and tensors
        self.corresponder.configure(initial_params)

        # Set up the automatic step size adapter
        self.step_size_adapter = DualAveragingStepSize(self.step_size)

        # Set the step counter to 0
        self._step_count = 0

    # Sample the momentum variables from a standard normal
    def _sample_friction(self, sample_prefix):
        loc = torch.zeros(self.corresponder.total_size)
        scale = (2 * (self.C - self.B_hat) * self.step_size 
                 * torch.ones(self.corresponder.total_size))
        return self.corresponder.normal_sample(loc, scale, sample_prefix)

    # Computes orig + step * grad elementwise, where orig and grad are
    # dictionaries with the same keys
    def _step_position(self, orig, step, grad):   
        return {site:(orig[site] + step * grad[site]) for site in orig}
        
    # Computes orig + step * grad elementwise, where orig and grad are
    # dictionaries with the same keys
    # If with_friction adds additional update terms elementwise
    def _step_momentum(self, orig, step, grad):   
        if self.with_friction:
            f = self._sample_friction(f"q_{self._step_count}")
            return {site:(orig[site] + step * grad[site] - step * self.C * orig[site] + (step / self.step_size) * f[site]) for site in orig}
        else:
            return {site:(orig[site] + step * grad[site]) for site in orig}

    # Sample the momentum variables from a standard normal
    def sample_momentum(self, sample_prefix):
        loc = torch.zeros(self.corresponder.total_size)
        scale = torch.ones(self.corresponder.total_size)
        return self.corresponder.normal_sample(loc, scale, sample_prefix)

    # Get the potential function for a minibatch
    def get_potential_fn(self, subsample=False):
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
            
        _, potential_fn, _, _ = initialize_model(
            self.model,
            model_args,
            self.model_kwargs,
            initial_params=self._initial_params,
            scale_likelihood=self.data_size/batch_size
        )
        return potential_fn

    # Update the position one step
    def update_position(self, p, q, potential_fn, step_size):
        return self._step_position(q, self.step_size, p)

    # Update the momentum one step
    def update_momentum(self, p, q, potential_fn, step_size):
        grad_q, _ = potential_grad(potential_fn, q)
        return self._step_momentum(p, - step_size, grad_q)

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
        potential_fn = self.get_potential_fn(subsample=True)

        # Position variable
        q = q_current = params

        # Resample the momentum
        p = p_current = self.sample_momentum(f"q_{self._step_count}")

        ## Simulate Hamiltonian dynamics using leapfrog method
        # Half-step momentum
        p = self.update_momentum(p, q, potential_fn, self.step_size / 2)

        # Full-step position and momentum alternately
        for i in range(self.num_steps):
            q = self.update_position(p, q, potential_fn, self.step_size)
            if i < self.num_steps - 1:
                p = self.update_momentum(p, q, potential_fn, self.step_size)

        # Finally half-step momentum
        p = self.update_momentum(p, q, potential_fn, self.step_size / 2)
        
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

    def _compute_accept_prob(self, potential_fn, q_current, p_current, q, p):
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