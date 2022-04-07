Reproducing Stochastic Gradient Hamiltonian Monte Carlo
=======================================================

This is a group project for the Advanced Topics in Machine Learning course at
Oxford. We are reproducing the experiments in the paper "Stochastic Gradient 
Hamiltonian Monte Carlo" [[1]](https://arxiv.org/abs/1402.4102) by Tianqi Chen, 
Emily B. Fox and Carlos Guestrin.

We implemented the following algorithms to be used directly with Pyro - a universal probabilistic programming language (PPL) written in Python [[2]](https://pyro.ai/): Hamiltonian Monte Carlo (HMC), Stochastic Gradient Hamiltonian Monte Carlo (SGHMC), Stochastic Gradient with Langevin Dynamics (SGLD), Stochastic Gradient Descent (SGD), and SGD with Nesterov momentum, Stochastic Gradient No U-Turn Sampler (SGNUTS).

- [bnn/](.bnn/) reproducing MNIST classification with Bayesian neural networks (BNN) from [[1]](https://arxiv.org/abs/1402.4102), with some additional experiments and demos.
- [demo.ipynb](demo.ipynb) a simple demo that goes through the caveats of our implementation and to how get started using it.
- [examples/](.examples/) a series of examples that demonstrate some of the algorithms and other options.
- [experiments/](.experiments/) reproducing Figure 1, Figure 2 and Figure 3 from [[1]](https://arxiv.org/abs/1402.4102).
- [kernel/](.kernel/) contains our implementations of HMC, SGHMC, SGLD, SGD, SGNUTS.