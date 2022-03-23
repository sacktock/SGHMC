
import torch

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.ops.integrator import potential_grad
from util import initialize_model
from param_tensor_corresponder import ParamTensorCorresponder
from collections import OrderedDict

class SGD(MCMCKernel):
    """Stochastic Gradient Descent
    
    Parameters
    ----------
    subsample_positions : list, default [0] / 1st positional argument only
        Specifies which positional arguments of the model to subsample during runtime
        
    batch_size : int, default=5
        The size of the minibatches to use

    step_size : int, default 0.1
        The size of a single step taken while simulating Hamiltonian dynamics

    num_steps : int, default 1
        The number of steps to simulate Hamiltonian dynamics

    with_momentum : bool, default=False
        Use Nesterov's momentum during parameter updates

    alpha : float, default=True
        Momentum hyperparameter indicting how much to weight momentum in favor of gradient
    """

    def __init__(self,
                 model, 
                 subsample_positions=[0], 
                 batch_size=5, 
                 step_size=0.1, 
                 num_steps=1,
                 with_momentum=True,
                 alpha=0.75):

        self.model = model
        self.subsample_positions = subsample_positions
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_steps = num_steps
        self.with_momentum = with_momentum
        self._initial_params = None
        self.corresponder = ParamTensorCorresponder()
        self.alpha = alpha

    def setup(self, warmup_steps, *model_args, **model_kwargs):

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

        initial_params, potential_fn, transforms, _ = initialize_model(
            self.model,
            self.model_args,
            self.model_kwargs,
            initial_params = self._initial_params
        )

        # Set up the corresponder between parameter dicts and tensors
        self.corresponder.configure(initial_params)

        # Initialise random parameters
        loc = torch.zeros(self.corresponder.total_size)
        scale = torch.ones(self.corresponder.total_size)
        initial_params = pyro.sample('initial_params', dist.Normal(loc, scale))
        initial_params = self.corresponder.to_params(initial_params)

        # Cache variables
        self._initial_params = initial_params
        self.full_potential_fn = potential_fn
        self.transforms = transforms

        self._momentum = self.corresponder.to_params(torch.zeros(self.corresponder.total_size))

        self._step_count = 0

    def _step_position(self, orig, targ):   
        return {site:(orig[site] + targ[site]) for site in orig}

    def update_position(self, x, v):
        return self._step_position(x, v)

    def _step_momentum(self, orig, grad):
        return {site : (orig[site] * self.alpha - self.step_size * grad[site]) for site in orig}

    def update_momentum(self, v, x, potential_fn):
        x_tilde = {site : (x[site] + self.alpha * v[site]) for site in x}
        grad, _  = potential_grad(potential_fn, x_tilde)
        return self._step_momentum(v, grad)

    def get_potential_fn(self, subsample=True):
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
            scale_likelihood=self.data_size/batch_size,
        )

        return potential_fn


    def sample(self, params):
        # Increment the step counter
        self._step_count += 1

        # Compute the new potential fn
        potential_fn = self.get_potential_fn(subsample=True)

        # Position variable
        x = params
        v = self._momentum
        
        for i in range(self.num_steps):
            if self.with_momentum:
                # Compute Nesterov momentum: v = self.alpha * v - self.step_size * grad_(x + self.alpha * v)
                v = self.update_momentum(v, x, potential_fn)
            else:
                # Compute v = - self.step_size * grad_x
                v = self.update_momentum(self._momentum, x, potential_fn)
            # Step position using momentum: x = x + v
            x = self.update_position(x, v)

        # Cache momentum
        if self.with_momentum:
            self._momentum = v

        return x

    def logging(self):
        return OrderedDict(
            [
                ("learning rate", "{:.2e}".format(self.step_size))
            ]
        )

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params
