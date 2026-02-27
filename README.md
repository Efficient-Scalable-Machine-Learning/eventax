<p align="center">
  <img src="./docs-site/docs/img/logo5.svg" alt="Eventpropjax" width="70%">
</p>

Eventax provides a [JAX](https://github.com/google/jax) implementation of the [EventProp algorithm](https://arxiv.org/abs/2009.08378) using [Diffrax](https://github.com/patrick-kidger/diffrax) and [Equinox](https://github.com/patrick-kidger/equinox) offering full autograd support, easy extension with custom neuron dynamics, and built-in delay training.

## Features
- Fully differentiable implementation via JAX and Diffrax
- Easy extension with custom neuron model dynamics + learnable parameters
- Support for (trainable) synnaptic delays.
- GPU/TPU compatibility through JAX

## 📦 Installation
```bash
pip install eventax
```
