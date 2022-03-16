import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from util import initialize_model
from pyro.ops.integrator import potential_grad

from base import StochasticMCMCKernel

class SGHMC(StochasticMCMCKernel):
    """Stochastic Gradient Hamiltonian Monte Carlo kernel.
    
    Parameters
    ----------
        
    model
        The pyro model from which to sample.

    step_size : int, default=1
        The size of a single step taken while simulating Hamiltonian dynamics

    num_steps : int, default 10
        The number of steps to simulate Hamiltonian dynamics
        
    with_friction : bool, default False
        Use friction term when updating momentum

    do_mh_correction : bool, default False
        compute the mh correction term using the whole dataset
    """

    def __init__(self, model, data, batch_size=5, step_size=1, num_steps=10,
                 with_friction=True, do_mh_correction=False):
        super().__init__(model,
            data,
            batch_size,
            step_size,
            num_steps, 
            do_mh_correction)
        self.with_friction = with_friction
        self._initial_params = None
        self.C = 1
        self.B_hat = 0

    def setup(self, warmup_steps, *model_args, **model_kwargs):

        # Compute the initial parameter and potential function from the model
        # Use entire dataset to find initial parameters using pyros in-built search
        # First model argument must be the dataset !!!!
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        try:
            batch_size = self.model_args[0].size(0)
        except AttributeError:
            raise RuntimeError('Could not compute get the size of the dataset - \
                    please make sure the first model argument is a pytorch tensor representing the dataset')
        initial_params, potential_fn, transforms, _ = initialize_model(
            self.model,
            self.model_args,
            self.model_kwargs,
            initial_params = self._initial_params,
            scale_likelihood = batch_size/self.data_size
        )
        self._initial_params = initial_params
        self.potential_fn = potential_fn
        self.transforms = transforms

        # Compute the dimension of each model parameter
        self._param_sizes = {}
        for site, values in self._initial_params.items():
            size = values.numel()
            self._param_sizes[site] = size

        # Set the step counter to 0
        self._step_count = 0

    # Sample the momentum variables from a standard normal
    def _sample_friction(self, sample_prefix):
        f = {}
        for site, size in self._param_sizes.items():
            f[site] = pyro.sample(
                f"{sample_prefix}_{site}",
                dist.Normal(torch.zeros(size), 2*(self.C - self.B_hat)*self.step_size*torch.ones(size))
            )
        return f

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
        p = {}
        for site, size in self._param_sizes.items():
            p[site] = pyro.sample(
                f"{sample_prefix}_{site}",
                dist.Normal(torch.zeros(size), torch.ones(size))
            )
        return p

    # Get the potential function for a minibatch
    def get_potential_fn(self, batch):
        _, potential_fn, _, _ = initialize_model(
            self.model,
            (batch,) + self.model_args[1:],
            self.model_kwargs,
            initial_params=self._initial_params,
            scale_likelihood=self.data_size/self.batch_size
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