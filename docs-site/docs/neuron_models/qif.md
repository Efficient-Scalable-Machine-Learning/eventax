# Quadratic Integrate and Fire (QIF)

Implementation of the Quadratic Integrate and Fire (QIF) neuron model in phase representation.
The QIF extends the [`LIF`][eventax.neuron_models.LIF] and inherits its spike handling ([`spike_condition`][eventax.neuron_models.LIF.spike_condition], [`input_spike`][eventax.neuron_models.LIF.input_spike], [`reset_spiked`][eventax.neuron_models.LIF.reset_spiked]) but replaces the voltage dynamics with quadratic ones.

The standard QIF dynamics in voltage space are:

\[
\tau_\text{mem} \frac{\partial V}{\partial t} = V^2 + I + I_c, \qquad
\tau_\text{syn} \frac{\partial I}{\partial t} = -I.
\]

To avoid the divergence of $V \to \infty$ at spike time, the model operates in **phase space** via the coordinate transform following [Klos and Memmersheimer](https://arxiv.org/abs/2309.14523):

\[
\varphi = \frac{1}{\pi}\arctan\!\left(\frac{V}{\pi}\right) + \frac{1}{2}, \qquad \varphi \in [0, 1].
\]

**Free Dynamics (phase representation)**

\[
\begin{aligned}
\tau_\text{mem} \frac{\partial \varphi}{\partial t} &= \cos(\pi\varphi)\left[\cos(\pi\varphi) + \frac{1}{\pi}\sin(\pi\varphi)\right] + \frac{1}{\pi^2}\sin^2(\pi\varphi)\,(I + I_c), \\
\tau_\text{syn} \frac{\partial I}{\partial t} &= -I.
\end{aligned}
\]

**Transition Condition**

\[
\varphi_n(t_s^-) - 1 = 0, \qquad \text{for any neuron } n.
\]

**Jumps at Transition**

\[
\begin{aligned}
\varphi_n(t_s^+) &= 0, \\
I_m\big(t_s^+\big) &= I_m\big(t_s^-\big) + W_{mn}, \quad \forall m.
\end{aligned}
\]

Where

 * $\varphi \in \mathbb{R}^N$ is the phase variable (bounded in $[0, 1]$),
 * $I \in \mathbb{R}^N$ is the synaptic current,
 * $I_c \in \mathbb{R}^N$ is the bias current,
 * $\tau_\text{syn}$ and $\tau_\text{mem}$ are the time constants,
 * $W \in \mathbb{R}^{(N+K) \times N}$ is the synaptic weight matrix with number of neurons $N$ and input size $K$.

The threshold and reset are fixed at $\vartheta = 1$ and $\varphi_\text{reset} = 0$ respectively.

::: eventax.neuron_models.QIF
    options:
      members: false

---

## Parameters

The QIF accepts the same parameters as the [`LIF`][eventax.neuron_models.LIF] except that `thresh`, `vreset`, and `reset_grad_preserve` are fixed internally and not exposed.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_neurons` | Number of neurons in the layer. | — |
| `in_size` | Number of input connections. | — |
| `tmem` | Membrane time constant $\tau_\text{mem}$. | — |
| `tsyn` | Synaptic time constant $\tau_\text{syn}$. | — |
| `wmask` | Binary mask for the weight matrix, shape $(N+K) \times N$. | — |
| `dtype` | JAX dtype for all computations. | `jnp.float32` |

### Weight initialisation

Same as [`LIF` — Weight initialisation](lif.md#weight-initialisation).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_weights` | If given, used directly as $W$. | `None` |
| `wmean` | Mean for random weight initialisation. | `0.0` |
| `wlim` | Uniform half-range for random weights. | `None` |
| `fan_in_mode` | Fan-in scaling mode. | `None` |
| `init_bias` | If given, used directly as $I_c$. | `None` |
| `bmean` | Mean for random bias initialisation. | `0.0` |
| `blim` | Uniform half-range for random biases. | `None` |

---

## State layout

The state array `y` has shape `(n_neurons, 2)`:

| Channel | Index | Description |
|---------|-------|-------------|
| $\varphi$ | `y[:, 0]` | Phase variable (bounded in $[0, 1]$) |
| $I$ | `y[:, 1]` | Synaptic current |

Initial state: $\varphi = 0.5$, $I = 0$.

!!! note
    The internal state uses the phase variable $\varphi$, not the voltage $V$. Use [`observe`][eventax.neuron_models.QIF.observe] to convert back to voltage space.

---

## Trainable fields

| Field | Shape | Description |
|-------|-------|-------------|
| `weights` | $(N+K) \times N$ | Connection weight matrix $W$ |
| `ic` | $(N,)$ | Bias current $I_c$ |

---

## Methods

### `init_state`

::: eventax.neuron_models.QIF.init_state
    options:
      show_root_heading: false

Returns state of shape `(n_neurons, 2)` with $\varphi = 0.5$ (corresponding to $V = 0$) and $I = 0$.

---

### `dynamics`

::: eventax.neuron_models.QIF.dynamics
    options:
      show_root_heading: false

Implements the QIF free dynamics in phase space. See the equations at the top of this page.

---

### `observe`

::: eventax.neuron_models.QIF.observe
    options:
      show_root_heading: false

Maps the phase variable back to voltage space via the inverse transform:

\[
V = \pi\tan\!\bigl(\pi(\varphi - \tfrac{1}{2})\bigr)
\]

Returns an array of shape `(n_neurons, 2)` where channel 0 is now the voltage $V$ instead of the phase $\varphi$, and channel 1 is the synaptic current $I$ (unchanged).

---

### Inherited methods

The following methods are inherited from [`LIF`][eventax.neuron_models.LIF] without modification:

- [`spike_condition`][eventax.neuron_models.LIF.spike_condition] — triggers when $\varphi$ crosses $\vartheta = 1$.
- [`input_spike`][eventax.neuron_models.LIF.input_spike] — adds $W_{n,m}$ to the synaptic current of target neurons.
- [`reset_spiked`][eventax.neuron_models.LIF.reset_spiked] — resets $\varphi$ to $0$ via `jnp.where` (`reset_grad_preserve=False`).
