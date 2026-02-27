# Izhikevich Neuron

Implementation of the [**Izhikevich spiking neuron model**](https://www.izhikevich.org/publications/spikes.htm) with a current-based exponential synapse.  
The model combines the two-dimensional neuron dynamics of Izhikevich (2003) with an exponentially decaying synaptic current channel, providing smooth temporal integration of incoming spikes.

---

**Free Dynamics**

\[
\begin{aligned}
\frac{\partial V}{\partial t} &= 0.04\,V^2 + 5\,V + 140 - U + I_c + I, \\[4pt]
\frac{\partial U}{\partial t} &= a\,(b\,V - U), \\[4pt]
\tau_\text{syn}\,\frac{\partial I}{\partial t} &= -I.
\end{aligned}
\]

The voltage equation captures the spike-generating dynamics via a quadratic nonlinearity.  
The recovery variable \(U\) provides slow negative feedback (e.g.\ K\(^+\) channel activation).  
The synaptic current \(I\) integrates incoming spikes and decays exponentially with time constant \(\tau_\text{syn}\).

**Transition Condition**

\[
\begin{aligned}
V_n(t_s^-) - \vartheta &= 0, \\
\dot{V}_n(t_s^-) &\neq 0, \\
\text{for any neuron } n.
\end{aligned}
\]

**Jumps at Transition**

When neuron \(n\) emits a spike at time \(t_s\), its **postsynaptic targets** \(m\) receive a current impulse:

\[
I_m\big(t_s^+\big) = I_m\big(t_s^-\big) + W_{nm}, \quad \forall m.
\]

The spiking neuron \(n\) itself undergoes a reset:

\[
\begin{aligned}
V_n(t_s^+) &= c, \\
U_n(t_s^+) &= U_n(t_s^-) + d.
\end{aligned}
\]

Times immediately before a jump are marked with \((-)\), and immediately after with \((+)\).

Where

 * \(V \in \mathbb{R}^N\) — membrane potential of the neurons,
 * \(U \in \mathbb{R}^N\) — recovery variable,
 * \(I \in \mathbb{R}^N\) — synaptic current,
 * \(I_c \in \mathbb{R}^N\) — bias current,
 * \(a, b, c, d\) — dimensionless parameters controlling the neuron type (see Izhikevich, 2003),
 * \(\tau_\text{syn}\) — synaptic time constant,
 * \(\vartheta\) — firing threshold applied to \(V\),
 * \(W \in \mathbb{R}^{(N+K) \times N}\) — synaptic weight matrix with number of neurons \(N\) and input size \(K\).

::: eventax.neuron_models.Izhikevich
    options:
      members: false

---

## Parameters

The Izhikevich parameters (`a`, `b`, `c`, `d`) accept either a scalar (shared across all neurons) or an array of shape `(n_neurons,)` for per-neuron values, enabling heterogeneous populations (e.g.\ mixing regular spiking and fast spiking neurons).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_neurons` | Number of neurons in the layer. | — |
| `in_size` | Number of input connections. | — |
| `a` | Recovery time scale $a$. Smaller values yield slower recovery. | `0.02` |
| `b` | Recovery sensitivity $b$. Controls coupling of $U$ to $V$. | `0.2` |
| `c` | Post-spike reset voltage $c$. | `-51.0` |
| `d` | Post-spike recovery increment $d$. | `2.0` |
| `v_thresh` | Spike threshold $\vartheta$. | `30.0` |
| `tau_syn` | Synaptic time constant $\tau_\text{syn}$. | `5.0` |
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
| `reset_grad_preserve` | If `True`, the reset is implemented as $V = V - (\vartheta - c)$, which preserves gradients through the reset. If `False`, uses `jnp.where` which blocks the gradient. | `True` |

---

## State layout

The state array `y` has shape `(n_neurons, 3)`:

| Channel | Index | Description |
|---------|-------|-------------|
| $V$ | `y[:, 0]` | Membrane voltage |
| $U$ | `y[:, 1]` | Recovery variable |
| $I$ | `y[:, 2]` | Synaptic current |

Initial state: $V_0 = c$, $U_0 = b \cdot c$, $I_0 = 0$.

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

::: eventax.neuron_models.Izhikevich.init_state
    options:
      show_root_heading: false

Returns state of shape `(n_neurons, 3)` with $V = c$, $U = b \cdot c$, $I = 0$.

---

### `dynamics`

::: eventax.neuron_models.Izhikevich.dynamics
    options:
      show_root_heading: false

Implements the free dynamics equations above. The voltage $V$ evolves via the characteristic quadratic nonlinearity with recovery feedback from $U$ and drive from the synaptic current $I$ and bias $I_c$. The recovery variable $U$ relaxes toward $b \cdot V$ with rate $a$. The synaptic current $I$ decays exponentially with time constant $\tau_\text{syn}$.

---

### `spike_condition`

::: eventax.neuron_models.Izhikevich.spike_condition
    options:
      show_root_heading: false

Returns $V - \vartheta$. A spike is triggered when this changes sign (crosses zero from below).

---

### `input_spike`

::: eventax.neuron_models.Izhikevich.input_spike
    options:
      show_root_heading: false

Adds the connection weights $W_{n,m}$ to the synaptic current channel of the target neurons. Only entries where `valid_mask` is `True` are applied.

---

### `reset_spiked`

::: eventax.neuron_models.Izhikevich.reset_spiked
    options:
      show_root_heading: false

Resets voltage and increments recovery for spiked neurons. The behaviour depends on `reset_grad_preserve`:

- **`True`** (default): $V = V - (\vartheta - c)$ and $U = U + d$. Preserves the gradient.
- **`False`**: $V = c$ via `jnp.where`, which blocks the gradient through the reset; $U = U + d$ unchanged.

The synaptic current $I$ is left unchanged by the reset.
