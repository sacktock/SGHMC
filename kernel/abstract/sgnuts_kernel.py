import sys
sys.path.append("..")

import torch
import pyro

from pyro.ops.integrator import potential_grad
import pyro.distributions as dist

from kernel.utils.main import initialize_model
from kernel.utils.param_tensor_corresponder import ParamTensorCorresponder
from kernel.utils.dual_averaging_step_size import DualAveragingStepSize

from kernel.legacy.sghmc import SGHMC

class SGHMC_for_NUTS(SGHMC):
    """Stochastic Gradient Hamiltonian Monte Carlo kernel implementation for NUTS.
    
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

    friction_term : dict or None, default=None
        The friction term to use for the Langevin dynamics. This should be a 
        dictionary of square tensors, one for each of the model parameters,
        with the size of each matrix determined by the dimension of each
        parameter. The default is to use identity matrices for each
        parameter.

    obs_info_noise : bool, default=True
        Use the observed information to estimate the noise model

    do_mh_correction : bool, default False
        Compute the mh correction term using the whole dataset
        
    do_step_size_adaptation : bool, default True
        Do step size adaptation during warm up phase

    Limitations
    -----------

    - Currently the stepsize adaptation algorithm used doesn't work well with 
    NUTS. Should use a different one...
    - `friction_term` must be constant, and can't vary according to the values
    of the parameters.
    - `friction_term` yeilds a block matrix: friction is applied independently
    to each parameter. One parameter's values can't affect the friction on
    another.
    """

    def __init__(self, model, subsample_positions=[0], batch_size=5, step_size=0.1, num_steps=10,
                 resample_every_n=50, with_friction=True, friction_term=None, friction_constant=1.0,
                 obs_info_noise=True, compute_obs_info='every_sample', do_mh_correction=False, 
                 do_step_size_adaptation=True, target_accept=0.8):
      
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
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None
        
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

        # Set up the automatic step size adapter
        self.step_size_adapter = DualAveragingStepSize(self.step_size)

        # Set the step counter to 0
        self._step_count = 0

        # Caches initial parameters, potential_energy and gradients
        z = {k: v.detach() for k, v in self.initial_params.items()}
        z_grads, potential_energy = potential_grad(self.full_potential_fn, z)
        self._cache(self.initial_params, potential_energy, z_grads)

    def _step_position(self, orig, mom): 
        try:
            return {site:(orig[site] + self.step_size * mom[site].view(orig[site].shape)) for site in orig}
        except:
            return super()._step_position(orig, mom)

    def _step_momentum(self, orig, grad):   
        try:
            if self.with_friction:
                fric = self._sample_friction(f"f_{self._step_count}")
                return {site:(orig[site].view(grad[site].shape) - self.step_size * grad[site] - abs(self.step_size) * self.C * orig[site].view(grad[site].shape) + fric[site].view(grad[site].shape)) for site in orig}
            else:
                return {site:(orig[site].view(grad[site].shape) - abs(self.step_size) * grad[site]) for site in orig}
        except:
            super()._step_momentum(orig, grad)
            

    def _cache(self, z, potential_energy, z_grads=None):
        self._z_last = z
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last, self._z_grads_last
