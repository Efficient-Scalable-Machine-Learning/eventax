# Leaky Integrate and Fire (LIF)
Implementation of the Leaky Integrate and Fire (LIF) neuron model.  
The model follows the following internal dynamics:

**Free Dynamics**

\[
\begin{aligned}
\tau_\text{mem} \frac{\partial V}{\partial t} &= -V + I + I_c, \\ 
\tau_\text{syn} \frac{\partial I}{\partial t} &= -I.
\end{aligned}
\]

**Transition Condition**

\[
\begin{aligned}
V_n(t_s^-) - \vartheta &= 0, \\
\dot{V}_n(t_s^-) &\neq 0, \\
\text{for any neuron } n.
\end{aligned}
\]

**Jumps at Transition**

\[
\begin{aligned}
V_n(t_s^+) &= V_\text{reset}, \\
I_m\big(t_s^+\big) &= I_m\big(t_s^-\big) + W_{mn}, \quad \forall m.
\end{aligned}
\]

Where

 * $I \in \mathbb{R}^N$ and \(V \in \mathbb{R}^N\) are the synaptic current and membrane potential of the neurons,
 * \(I_c \in \mathbb{R}^N\) is the bias current,
 * \(\tau_\text{syn}\) and \(\tau_\text{mem}\) are the time constants of those channels respectively,
 * \(\vartheta\) is the threshold potential,
 * \(W \in \mathbb{R}^{(N+K) \times N}\) is the synaptic weight matrix with number of neurons $N$ and input size $K$,

Times immediately before a jump are marked with \(-\), and immediately after with \(+\).

::: eventax.neuron_models.LIF
    options:
      members: false

---

## Parameters

All neuron-level parameters (`thresh`, `tmem`, `tsyn`, `vreset`) accept either a scalar (shared across all neurons) or an array of shape `(n_neurons,)` for per-neuron values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_neurons` | Number of neurons in the layer. | — |
| `in_size` | Number of input connections. | — |
| `thresh` | Spike threshold $\vartheta$. | — |
| `tmem` | Membrane time constant $\tau_\text{mem}$. | — |
| `tsyn` | Synaptic time constant $\tau_\text{syn}$. | — |
| `vreset` | Reset voltage $V_\text{reset}$. | `0` |
| `wmask` | Binary mask for the weight matrix, shape $(N+K) \times N$. | — |
| `dtype` | JAX dtype for all computations. | `jnp.float32` |

### Weight initialisation

Weights and biases are initialised via [`init_weights_and_bias`][eventax.neuron_models.initializations.init_weights_and_bias].

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_weights` | If given, used directly as $W$. | `None` |
| `wmean` | Mean for random weight initialisation. | `0.0` |
| `wlim` | Uniform half-range for random weights. | `None` |
| `fan_in_mode` | Fan-in scaling mode. | `None` |
| `init_bias` | If given, used directly as $I_c$. | `None` |
| `bmean` | Mean for random bias initialisation. | `0.0` |
| `blim` | Uniform half-range for random biases. | `None` |

### Gradient behaviour

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reset_grad_preserve` | If `True`, the reset is implemented as $V = V - (\vartheta - V_\text{reset})$, which preserves gradients through the reset. If `False`, uses `jnp.where` which blocks the gradient. | `True` |

---

## State layout

The state array `y` has shape `(n_neurons, 2)`:

| Channel | Index | Description |
|---------|-------|-------------|
| $V$ | `y[:, 0]` | Membrane voltage |
| $I$ | `y[:, 1]` | Synaptic current |

Initial state is all zeros.

---

## Trainable fields

| Field | Shape | Description |
|-------|-------|-------------|
| `weights` | $(N+K) \times N$ | Connection weight matrix $W$ |
| `ic` | $(N,)$ | Bias current $I_c$ |

All other fields are static (not optimised by gradient-based training).

---

## Methods

### `init_state`

::: eventax.neuron_models.LIF.init_state
    options:
      show_root_heading: false

Returns zeros of shape `(n_neurons, 2)`.

---

### `dynamics`

::: eventax.neuron_models.LIF.dynamics
    options:
      show_root_heading: false

Implements the free dynamics ODE above.

---

### `spike_condition`

::: eventax.neuron_models.LIF.spike_condition
    options:
      show_root_heading: false

Returns $V - \vartheta$. A spike is triggered when this changes sign (crosses zero from below).

---

### `input_spike`

::: eventax.neuron_models.LIF.input_spike
    options:
      show_root_heading: false

Adds the connection weights $W_{n,m}$ to the synaptic current channel of the target neurons. Only entries where `valid_mask` is `True` are applied.

---

### `reset_spiked`

::: eventax.neuron_models.LIF.reset_spiked
    options:
      show_root_heading: false

Resets the voltage of spiked neurons. The behaviour depends on `reset_grad_preserve`:

- **`True`** (default): $V = V - (\vartheta - V_\text{reset})$. Preserves the gradient.
- **`False`**: $V = V_\text{reset}$ via `jnp.where`, which blocks the gradient through the reset.
