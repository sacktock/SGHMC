# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import biject_to

import pyro
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.autoguide.initialization import InitMessenger, init_to_uniform
from pyro.poutine.subsample_messenger import _Subsample

from pyro.infer.mcmc.util import (
    TraceEinsumEvaluator, 
    _guess_max_plate_nesting,
    _PEMaker,
    _find_valid_initial_params
)


class _PEMakerScale(_PEMaker):
    def __init__(
        self,
        model,
        model_args,
        model_kwargs,
        trace_prob_evaluator,
        transforms,
        observation_nodes,
        scale_likelihood
    ):
        super().__init__(
            model,
            model_args,
            model_kwargs,
            trace_prob_evaluator,
            transforms,
        )
        self.observation_nodes = observation_nodes
        self.scale_likelihood = scale_likelihood

    def _potential_fn(self, params):
        params_constrained = {k: self.transforms[k].inv(v) for k, v in params.items()}
        cond_model = poutine.condition(self.model, params_constrained)
        model_trace = poutine.trace(cond_model).get_trace(
            *self.model_args, **self.model_kwargs
        )
        if self.scale_likelihood != 1.0:
            for node in self.observation_nodes:
                model_trace.nodes[node]['scale'] = self.scale_likelihood
        log_joint = self.trace_prob_evaluator.log_prob(model_trace)
        for name, t in self.transforms.items():
            log_joint = log_joint - torch.sum(
                t.log_abs_det_jacobian(params_constrained[name], params[name])
            )
        return -log_joint

def initialize_model(
    model,
    model_args=(),
    model_kwargs={},
    transforms=None,
    max_plate_nesting=None,
    jit_compile=False,
    jit_options=None,
    skip_jit_warnings=False,
    num_chains=1,
    init_strategy=init_to_uniform,
    initial_params=None,
    scale_likelihood=1.0,
):
    """
    Given a Python callable with Pyro primitives, generates the following model-specific
    properties needed for inference using HMC/NUTS kernels:

    - initial parameters to be sampled using a HMC kernel,
    - a potential function whose input is a dict of parameters in unconstrained space,
    - transforms to transform latent sites of `model` to unconstrained space,
    - a prototype trace to be used in MCMC to consume traces from sampled parameters.

    :param model: a Pyro model which contains Pyro primitives.
    :param tuple model_args: optional args taken by `model`.
    :param dict model_kwargs: optional kwargs taken by `model`.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param int num_chains: Number of parallel chains. If `num_chains > 1`,
        the returned `initial_params` will be a list with `num_chains` elements.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain.
    :returns: a tuple of (`initial_params`, `potential_fn`, `transforms`, `prototype_trace`)
    """
    # XXX `transforms` domains are sites' supports
    # FIXME: find a good pattern to deal with `transforms` arg
    if transforms is None:
        automatic_transform_enabled = True
        transforms = {}
    else:
        automatic_transform_enabled = False
    if max_plate_nesting is None:
        max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)
    # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
    # No-op if model does not have any discrete latents.
    model = poutine.enum(
        config_enumerate(model), first_available_dim=-1 - max_plate_nesting
    )
    prototype_model = poutine.trace(InitMessenger(init_strategy)(model))
    model_trace = prototype_model.get_trace(*model_args, **model_kwargs)
    has_enumerable_sites = False
    prototype_samples = {}
    for name, node in model_trace.iter_stochastic_nodes():
        fn = node["fn"]
        if isinstance(fn, _Subsample):
            if fn.subsample_size is not None and fn.subsample_size < fn.size:
                raise NotImplementedError(
                    "HMC/NUTS does not support model with subsample sites."
                )
            continue
        if node["fn"].has_enumerate_support:
            has_enumerable_sites = True
            continue
        # we need to detach here because this sample can be a leaf variable,
        # so we can't change its requires_grad flag to calculate its grad in
        # velocity_verlet
        prototype_samples[name] = node["value"].detach()
        if automatic_transform_enabled:
            transforms[name] = biject_to(node["fn"].support).inv

    trace_prob_evaluator = TraceEinsumEvaluator(
        model_trace, has_enumerable_sites, max_plate_nesting
    )

    observation_nodes = model_trace.observation_nodes

    pe_maker = _PEMakerScale(
        model, 
        model_args, 
        model_kwargs, 
        trace_prob_evaluator, 
        transforms, 
        observation_nodes,
        scale_likelihood
    )

    if initial_params is None:
        prototype_params = {k: transforms[k](v) for k, v in prototype_samples.items()}
        # Note that we deliberately do not exercise jit compilation here so as to
        # enable potential_fn to be picklable (a torch._C.Function cannot be pickled).
        # We pass model_trace merely for computational savings.
        initial_params = _find_valid_initial_params(
            model,
            model_args,
            model_kwargs,
            transforms,
            pe_maker.get_potential_fn(),
            prototype_params,
            num_chains=num_chains,
            init_strategy=init_strategy,
            trace=model_trace,
        )
    potential_fn = pe_maker.get_potential_fn(
        jit_compile, skip_jit_warnings, jit_options
    )
    return initial_params, potential_fn, transforms, model_trace