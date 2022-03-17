Reproducing Stochastic Gradient Hamiltonian Monte Carlo
=======================================================

This is a group project for the Advanced Topics in Machine Learning course at
Oxford. We are reproducing the experiments in the paper [Stochastic Gradient 
Hamiltonian Monte Carlo](https://arxiv.org/abs/1402.4102) by Tianqi Chen, 
Emily B. Fox and Carlos Guestrin.

We are integrating our implementation of the algorithms in Pyro.

- [sghmc.py](sghmc.py) contains our implementation of the Stochastic Gradient 
Hamiltonian Monte Carlo (SGHMC) algorithm.
- [demo.ipynb](demo.ipynb) a simple demo that goes through the caveats of our implementation of SGHMC and to how get started using it.
- [example.ipynb](example.ipynb) a beta prior, binomial likelihood coin flip example that shows our implementation of SGHMC sampling from the posterior distribution. 
