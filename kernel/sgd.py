import sys
sys.path.append("..")

import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.ops.integrator import potential_grad
from kernel.utils.main import initialize_model
from kernel.utils.param_tensor_corresponder import ParamTensorCorresponder
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
        Analagous to learning rate

    weight_decay: float, default 0.0
        L2 weight penalisation

    with_momentum : bool, default=False
        Use Nesterov's momentum during parameter updates

    alpha : float, default=True
        Momentum hyperparameter indicting how much to weight momentum in favor of gradient
    """

    def __init__(self,
                 model, 
                 subsample_positions=[0], 
                 batch_size=5, 
                 learning_rate=0.1, 
                 weight_decay=0.0,
                 with_momentum=True,
                 momentum_decay=0.75):

        self.model = model
        self.subsample_positions = subsample_positions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.with_momentum = with_momentum
        self._initial_params = None
        self.corresponder = ParamTensorCorresponder()
        self.momentum_decay = momentum_decay
        self._dampening = 0.0
        self._momentum = None

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

        if self._initial_params is None:
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

        self.corresponder.configure(self._initial_params)

        # Initialise random parameters
        #loc = torch.zeros(self.corresponder.total_size)
        #scale = torch.ones(self.corresponder.total_size)
        #initial_params = pyro.sample('initial_params', dist.Normal(loc, scale))
        #initial_params = self.corresponder.to_params(initial_params)

        self._step_count = 0

    def _step_position(self, x, grad):   
        return {site:(x[site] - self.learning_rate * grad[site]) for site in x}

    def update_position(self, x, grad):
        return self._step_position(x, grad)

    def _step_grad(self, grad, v):
        return {site:(grad[site] + (1 - self.momentum_decay) * v[site].view(grad[site].shape)) for site in grad}

    def update_grad(self, grad, v):
        return self._step_grad(grad, v)

    def _step_momentum(self, orig, grad):
        if self._momentum is not None: 
            return {site : ((1 - self.momentum_decay) * orig[site].view(grad[site].shape) + (1 - self._dampening) * grad[site]) for site in orig}
        else:
            return {site : ((1 - self.momentum_decay) * orig[site].view(grad[site].shape) +  grad[site]) for site in orig}

    def _add_weight_decay(self, grad, x):
        return {site : (self.learning_rate * (grad[site] + self.weight_decay * x[site])) for site in grad}

    def update_momentum(self, v, grad):
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
        
        grad, _  = potential_grad(potential_fn, x)
        if self.weight_decay != 0.0:
            grad = self._add_weight_decay(grad, x)

        if self._momentum is not None:
            v = self._momentum
        else:
            v = self._momentum = self.corresponder.to_params(torch.zeros(self.corresponder.total_size))

        if self.with_momentum:
            v = self.update_momentum(v, grad)
            grad = self.update_grad(grad, v)

        # Step position using momentum: x = x + v
        x = self.update_position(x, grad)

        # Cache momentum
        if self.with_momentum:
            self._momentum = v

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
