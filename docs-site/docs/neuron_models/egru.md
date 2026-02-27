# Event-based Gated Recurrent Unit (EGRU)

Implementation of the [**Event-based Gated Recurrent Unit (EGRU)**](https://arxiv.org/abs/2206.06178) model.  
This neuron model combines exponential relaxation dynamics with gated recurrent interactions triggered by discrete events (spikes).  
It generalizes recurrent neural units to the event-driven setting and provides continuous-time dynamics between spikes.

---

**Free Dynamics**

Let

\[
u = \sigma(a_u), \qquad r = \sigma(a_r), \qquad z = \tanh(a_z),
\]

where \(\sigma(\cdot)\) is the logistic sigmoid.  
Each neuron maintains a continuous cell state \(c\) and three exponentially relaxing pre-activation variables \(a_u, a_r, a_z\):

\[
\begin{aligned}
\tau_\text{mem}\,\frac{\partial c}{\partial t} &= u\,(z - c), \\[4pt]
\tau_\text{syn}\,\frac{\partial a_u}{\partial t} &= -a_u + b_u,\\
\tau_\text{syn}\,\frac{\partial a_r}{\partial t} &= -a_r + b_r,\\
\tau_\text{syn}\,\frac{\partial a_z}{\partial t} &= -a_z + b_z.
\end{aligned}
\]

These equations describe exponentially decaying internal activations and a gated relaxation of the state \(c\) toward the modulation signal \(z\).

**Transition Condition**

A spike (event) is emitted whenever the continuous state \(c_n\) of neuron \(n\) reaches threshold:

\[
\begin{aligned}
c_n(t_s^-) - \vartheta &= 0, \\
\dot{c}_n(t_s^-) &\neq 0, \\
\text{for any neuron } n.
\end{aligned}
\]

**Jumps at Transition**

When neuron \(n\) emits a spike at time \(t_s\), its **postsynaptic targets** \(m\) experience instantaneous jumps:

\[
\begin{aligned}
a_{u,m}\big(t_s^+\big)
  &= a_{u,m}\big(t_s^-\big)
   + W^{(u)}_{mn}\, c_n(t_s^-), \\[4pt]
a_{r,m}\big(t_s^+\big)
  &= a_{r,m}\big(t_s^-\big)
   + W^{(r)}_{mn}\, c_n(t_s^-), \\[4pt]
a_{z,m}\big(t_s^+\big)
  &= a_{z,m}\big(t_s^-\big)
   + W^{(z)}_{mn}\, r_n(t_s^-)\, c_n(t_s^-).
\end{aligned}
\]

Here \(r_n = \sigma(a_{r,n})\) is the **presynaptic reset gate**, controlling the contribution of neuron \(n\)'s spikes to the \(a_z\) channel of its targets.

The spiking neuron \(n\) itself resets its internal state:

\[
c_n(t_s^+) = 0.
\]

Times immediately before a jump are marked with \((-)\), and immediately after with \((+)\).

Where

 * \(c \in \mathbb{R}^N\) — continuous cell state of the neurons,
 * \(a_u, a_r, a_z \in \mathbb{R}^N\) — pre-activations for the update, reset, and modulation gates,
 * \(u=\sigma(a_u)\), \(r=\sigma(a_r)\), \(z=\tanh(a_z)\) — corresponding gate activations,
 * \(b_u, b_r, b_z \in \mathbb{R}^N\) — bias levels (exponential attractors) for the pre-activations,
 * \(\tau_\text{mem}\), \(\tau_\text{syn}\) — membrane and synaptic time constants,
 * \(W^{(u)}, W^{(r)}, W^{(z)} \in \mathbb{R}^{(N+K)\times N}\) — synaptic weight matrices for the three channels,
 * \(\vartheta\) — firing threshold applied to \(c\),
 * \(N\) — number of neurons, \(K\) — number of input channels.

::: eventax.neuron_models.EGRU
    options:
      members: false

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_neurons` | Number of neurons in the layer. | — |
| `in_size` | Number of input connections. | — |
| `tmem` | Membrane time constant $\tau_\text{mem}$. | `20.0` |
| `tsyn` | Synaptic time constant $\tau_\text{syn}$. | `5.0` |
| `thresh` | Spike threshold $\vartheta$ on cell state $c$. | `0.5` |
| `wmask` | Binary mask for the weight matrices, shape $(N+K) \times N$. | — |
| `dtype` | JAX dtype for all computations. | `jnp.float32` |

### Initialisation

Weights and biases are drawn from normal distributions controlled by scale and mean parameters.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_scale` | Standard deviation for weight initialisation. | `5.0` |
| `weight_mean` | Mean for weight initialisation. | `1.0` |
| `bias_scale` | Standard deviation for bias initialisation. | `0.1` |
| `bias_mean` | Mean for bias initialisation. | `0.4` |

---

## State layout

The state array `y` has shape `(n_neurons, 4)`:

| Channel | Index | Description |
|---------|-------|-------------|
| $c$ | `y[:, 0]` | Cell state |
| $a_u$ | `y[:, 1]` | Update gate pre-activation |
| $a_r$ | `y[:, 2]` | Reset gate pre-activation |
| $a_z$ | `y[:, 3]` | Modulation gate pre-activation |

Initial state is all zeros.

---

## Trainable fields

| Field | Shape | Description |
|-------|-------|-------------|
| `W_u` | $(N+K) \times N$ | Weight matrix $W^{(u)}$ for the update gate |
| `W_r` | $(N+K) \times N$ | Weight matrix $W^{(r)}$ for the reset gate |
| `W_z` | $(N+K) \times N$ | Weight matrix $W^{(z)}$ for the modulation gate |
| `b_u` | $(N,)$ | Bias attractor $b_u$ for update pre-activation |
| `b_r` | $(N,)$ | Bias attractor $b_r$ for reset pre-activation |
| `b_z` | $(N,)$ | Bias attractor $b_z$ for modulation pre-activation |

All other fields are static (not optimised by gradient-based training).

---

## Methods

### `init_state`

::: eventax.neuron_models.EGRU.init_state
    options:
      show_root_heading: false

Returns zeros of shape `(n_neurons, 4)`.

---

### `dynamics`

::: eventax.neuron_models.EGRU.dynamics
    options:
      show_root_heading: false

Implements the free dynamics equations above. The pre-activations $a_u, a_r, a_z$ relax exponentially toward their bias attractors $b_u, b_r, b_z$ with time constant $\tau_\text{syn}$. The cell state $c$ evolves via gated relaxation toward $z = \tanh(a_z)$, modulated by the update gate $u = \sigma(a_u)$, with time constant $\tau_\text{mem}$.

---

### `spike_condition`

::: eventax.neuron_models.EGRU.spike_condition
    options:
      show_root_heading: false

Returns $c - \vartheta$. A spike is triggered when this changes sign (crosses zero from below).

---

### `input_spike`

::: eventax.neuron_models.EGRU.input_spike
    options:
      show_root_heading: false

Updates the pre-activation channels of target neurons when a spike arrives. The three channels receive different contributions:

- $a_u$: receives $W^{(u)}_{n,m}$
- $a_r$: receives $W^{(r)}_{n,m}$
- $a_z$: receives $W^{(z)}_{n,m} \cdot r_m$, gated by the **target** neuron's reset gate $r_m = \sigma(a_{r,m})$

Only entries where `valid_mask` is `True` are applied.

---

### `reset_spiked`

::: eventax.neuron_models.EGRU.reset_spiked
    options:
      show_root_heading: false

Resets the cell state of spiked neurons via subtraction: $c = c - \vartheta$. This preserves the gradient. The pre-activation channels $a_u, a_r, a_z$ are left unchanged.
