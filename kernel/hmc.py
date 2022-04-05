import sys
sys.path.append("..")

import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad
from kernel.utils.dual_averaging_step_size import DualAveragingStepSize
from collections import OrderedDict

class HMC(MCMCKernel):
    """Hamiltonian Monte Carlo kernel.
    
    Parameters
    ----------
        
    model
        The pyro model from which to sample.

    step_size : int, default=1
        The size of a single step taken while simulating Hamiltonian dynamics

    num_steps
        The number of steps to simulate Hamiltonian dynamics

    do_step_size_adaptation : bool, default True
        Do step size adaptation during warm up phase

    target_accept : int, default=0.8
        The target acceptance probability to aim for when doing step size 
        adaptation
    """

    def __init__(self, model, step_size=1, num_steps=10, do_step_size_adaptation=True, target_accept=0.8):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self.do_step_size_adaptation = do_step_size_adaptation
        self.target_accept = target_accept
        self._initial_params = None

    def setup(self, warmup_steps, *model_args, **model_kwargs):
        self._warmup_steps = warmup_steps

        # Compute the initial parameter and potential function from the model
        initial_params, potential_fn, transforms, _ = initialize_model(
            self.model,
            model_args,
            model_kwargs,
        )
        self._initial_params = initial_params
        self.potential_fn = potential_fn
        self.transforms = transforms

        # Compute the dimension of each model parameter
        self._param_sizes = {}
        for site, values in self._initial_params.items():
            size = values.numel()
            self._param_sizes[site] = size

        # Set up the automatic step size adapter
        self.step_size_adapter = DualAveragingStepSize(self.step_size, target_accept=self.target_accept)

        # Set the step counter to 0
        self._step_count = 0


    # Sample the momentum variables from a standard normal
    def _sample_momentum(self, sample_prefix):
        p = {}
        for site, size in self._param_sizes.items():
            p[site] = pyro.sample(
                f"{sample_prefix}_{site}",
                dist.Normal(torch.zeros(size), torch.ones(size))
            )
        return p

    # Computes orig + step * grad elementwise, where orig and grad are
    # dictionaries with the same keys
    def _step_variable(self, orig, step, grad):
        return {site:(orig[site] + step * grad[site]) for site in orig}

    # Compute the kinetic energy, given the momentum
    def _kinetic_energy(self, p):
        energy = torch.zeros(1)
        for site, value in p.items():
            energy += torch.dot(value, value)
        return 0.5 * energy

    def sample(self, params):

        # Increment the step counter
        self._step_count += 1

        # Position variable
        q = q_current = params

        # Resample the momentum
        p = p_current = self._sample_momentum(f"q_{self._step_count}")

        ## Simulate Hamiltonian dynamics using leapfrog method
        # Half-step momentum
        grad_q, _ = potential_grad(self.potential_fn, q)
        p = self._step_variable(p, - self.step_size / 2, grad_q)

        # Full-step position and momentum alternately
        for i in range(self.num_steps):
            q = self._step_variable(q, self.step_size, p)
            if i < self.num_steps - 1:
                grad_q, _ = potential_grad(self.potential_fn, q)
                p = self._step_variable(p, - self.step_size, grad_q)

        # Finally half-step momentum
        grad_q, _ = potential_grad(self.potential_fn, q)
        p = self._step_variable(p, - self.step_size / 2, grad_q)

        ## Metropolis-Hastings correction
        # Compute the current and proposed total energies
        energy_current = (self.potential_fn(q_current) 
                          + self._kinetic_energy(p_current))
        energy_proposal = self.potential_fn(q) + self._kinetic_energy(p)

        # Compute the acceptance probability
        energy_delta = energy_current - energy_proposal
        accept_prob = energy_delta.exp().clamp(max=1.0).item()

        # Update the step size with the step size adapter using the true acceptance prob
        if self._step_count <= self._warmup_steps and self.do_step_size_adaptation:
            self._update_step_size(accept_prob)

        # Draw a uniform random sample
        rand = pyro.sample(f"rand_{self._step_count}", dist.Uniform(0,1))

        # Accept the new point with probability `accept_prob`
        if rand < accept_prob:
            return q
        else:
            return q_current

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