from abc import ABCMeta, abstractmethod

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

class StochasticMCMCKernel(MCMCKernel, metaclass=ABCMeta):
    """Abstract base class for Stochastic MCMC kernels."""

    def __init__(self, model, data, batch_size=5, step_size=1, num_steps=10, 
                 do_mh_correction=False):
        self.model = model
        self.data = data
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_steps = num_steps
        self.do_mh_correction = do_mh_correction
        self._initial_params = None


    @abstractmethod
    def _sample_momentum(self):
        """Resample the momentum variables."""
        raise NotImplementedError

    @abstractmethod
    def _get_potential_fn(self, batch):
        """Calculates the potential function, given a minibatch."""
        raise NotImplementedError

    @abstractmethod
    def _update_position(self, p, q, potential_fn, step_size):
        """Update the position variables."""
        raise NotImplementedError

    @abstractmethod
    def _update_momentum(self, p, q, potential_fn, step_size):
        """Update the position variables."""
        raise NotImplementedError

    @abstractmethod
    def _kinetic_energy(self, p):
        """Compute the kinetic energy, given the momentum."""
        raise NotImplementedError

    def sample(self, params):

        # Increment the step counter
        self._step_count += 1
        
        # Sample a random mini batch
        perm = torch.randperm(self.data_size)
        idx = perm[:self.batch_size]
        batch = self.data[idx]

        # Compute the new potential fn
        potential_fn = self._get_potential_fn(batch)

        # Position variable
        q = q_current = params

        # Resample the momentum
        p = p_current = self._sample_momentum(f"q_{self._step_count}")

        ## Simulate Hamiltonian dynamics using leapfrog method
        # Half-step momentum
        p = self._update_momentum(p, q, potential_fn, self.step_size / 2)

        # Full-step position and momentum alternately
        for i in range(self.num_steps):
            q = self._update_position(p, q, potential_fn, self.step_size)
            if i < self.num_steps - 1:
                p = self._update_momentum(p, q, potential_fn, self.step_size)

        # Finally half-step momentum
        p = self._update_momentum(p, q, potential_fn, self.step_size / 2)
        
        ## Metropolis-Hastings correction
        if self.do_mh_correction:
            
            # Get the potential function for the whole dataset
            potential_fn = self._get_potential_fn(self.data)

            # Compute the current and proposed total energies
            energy_current = (self.potential_fn(q_current) 
                              + self._kinetic_energy(p_current))
            energy_proposal = self.potential_fn(q) + self._kinetic_energy(p)

            # Compute the acceptance probability
            energy_delta = energy_current - energy_proposal
            accept_prob = energy_delta.exp().clamp(max=1.0).item()

            # Draw a uniform random sample
            rand = pyro.sample(f"rand_{self._step_count}", dist.Uniform(0,1))

            # Accept the new point with probability `accept_prob`
            if rand < accept_prob:
                return q
            else:
                return q_current
        else:
            return q


    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params