import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from util import initialize_model
from pyro.ops.integrator import potential_grad

class SGHMC(MCMCKernel):
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

    def __init__(self, model, data, batch_size=5, step_size=1, num_steps=10, with_friction=True, do_mh_correction=False):
        self.model = model
        self.data = data
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_steps = num_steps
        self.do_mh_correction = do_mh_correction
        self.with_friction = with_friction
        self._initial_params = None
        self.C = 1
        self.B_hat = 0

    def setup(self, warmup_steps, *model_args, **model_kwargs):

        # Compute the initial parameter and potential function from the model
        initial_params, potential_fn, transforms, _ = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            initial_params = self._initial_params,
            scale_likelihood=self.data_size/self.batch_size
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
    def _sample_momentum(self, sample_prefix):
        p = {}
        for site, size in self._param_sizes.items():
            p[site] = pyro.sample(
                f"{sample_prefix}_{site}",
                dist.Normal(torch.zeros(size), torch.ones(size))
            )
        return p

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
    def _step_variable(self, orig, step, grad):   
        if self.with_friction:
            f = self._sample_friction(f"q_{self._step_count}")
            return {site:(orig[site] + step * grad[site] - step * self.C * orig[site] + (step / self.step_size) * f[site]) for site in orig}
        else:
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
        
        # Sample a random mini batch
        perm = torch.randperm(self.data_size)
        idx = perm[:self.batch_size]
        batch = self.data[idx]

        # Compute the new potential fn
        _, potential_fn, transforms, _ = initialize_model(
            self.model,
            model_args=(batch,),
            initial_params=self._initial_params,
            scale_likelihood=self.data_size/self.batch_size
        )

        self.potential_fn = potential_fn

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
        
        if self.do_mh_correction:
            ## Metropolis-Hastings correction
            # Compute the current and proposed total energies
            _, potential_fn, transforms, _ = initialize_model(
                self.model,
                model_args=(self.data,),
                initial_params=q,
                scale_likelihood=1.0
            )

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