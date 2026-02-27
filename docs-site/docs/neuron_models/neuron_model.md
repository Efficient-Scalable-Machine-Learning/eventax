# Neuron Models

A `NeuronModel` defines the behaviour of a neuron type within the event-driven simulation. It specifies:

- how state is **initialised** ([`init_state`][eventax.neuron_models.base_model.NeuronModel.init_state]),
- how state **evolves** between events ([`dynamics`][eventax.neuron_models.base_model.NeuronModel.dynamics]),
- **when** a neuron fires ([`spike_condition`][eventax.neuron_models.base_model.NeuronModel.spike_condition]),
- what happens when a spike **arrives** ([`input_spike`][eventax.neuron_models.base_model.NeuronModel.input_spike]),
- what happens when a neuron **itself fires** ([`reset_spiked`][eventax.neuron_models.base_model.NeuronModel.reset_spiked]).

All neuron models inherit from the abstract base class below.

---

::: eventax.neuron_models.base_model.NeuronModel
    options:
      members: false

---

## Methods

All of the following methods must be implemented by any concrete `NeuronModel` subclass. See [Examples: Neuron Models](../examples/neuron_models.ipynb) for a walkthrough of creating your own neuron model.

---

### `init_state`

::: eventax.neuron_models.base_model.NeuronModel.init_state
    options:
      show_root_heading: false

Should return the initial state for all neurons and their channels in the network. The returned array has shape `(n_neurons, n_channels)` where `n_channels` is determined by the specific neuron model (e.g. 2 for a LIF/QIF).

---

### `dynamics`

::: eventax.neuron_models.base_model.NeuronModel.dynamics
    options:
      show_root_heading: false

Defines the continuous-time dynamics of all neuron channels as an ODE. Given the current time $t$ and full state $y$ over all neurons, returns the derivative $\frac{\mathrm{d}y}{\mathrm{d}t}$.

**Arguments:**

- `t`: the current simulation time.
- `y`: the state of all neurons, shape `(n_neurons, n_channels)`.
- `args`: a dictionary of additional arguments (e.g. external input currents).

**Returns:**

The derivative of `y`, same shape as `y`.

---

### `spike_condition`

::: eventax.neuron_models.base_model.NeuronModel.spike_condition
    options:
      show_root_heading: false

Defines the condition under which a neuron generates a new event. Returns a vector over all neurons. An event for neuron $m$ is triggered when the value at index $m$ changes its sign from negative to positive from one ODE solver step to the next. A root finder then locates the exact event time.

!!! example

    For a simple threshold crossing at $V_\mathrm{th}$, this would return `y[:, 0] - V_th`.

**Arguments:**

- `t`: the current simulation time.
- `y`: the state of all neurons, shape `(n_neurons, n_channels)`.
- `args`: a dictionary of additional arguments.

**Returns:**

A vector of shape `(n_neurons,)`.

---

### `input_spike`

::: eventax.neuron_models.base_model.NeuronModel.input_spike
    options:
      show_root_heading: false

Defines how the internal state is updated when a spike is received. Must handle the case where multiple target neurons receive the spike simultaneously (i.e. `to_idx` may contain multiple indices).

**Arguments:**

- `y`: the current state of all neurons.
- `from_idx`: scalar index of the neuron that fired.
- `to_idx`: indices of the target neurons, shape `(targets,)`.
- `valid_mask`: boolean mask indicating which entries in `to_idx` are valid, shape `(targets,)`.

**Returns:**

The updated state `y`.

---

### `reset_spiked`

::: eventax.neuron_models.base_model.NeuronModel.reset_spiked
    options:
      show_root_heading: false

Defines how the state of a neuron is reset after it fires. Receives the current state and a one-hot mask indicating which neuron spiked.

**Arguments:**

- `y`: the current state of all neurons.
- `spiked_mask`: boolean array of shape `(n_neurons,)`, `True` for the neuron that spiked.

**Returns:**

The updated state `y`.

---

### `observe`

::: eventax.neuron_models.base_model.NeuronModel.observe
    options:
      show_root_heading: false

Extracts the observable channels from the full internal state. By default returns `y` unchanged. Override this in models where the internal state contains channels that should not be recorded (e.g. auxiliary solver variables).

**Returns:**

Array of shape `(n_neurons, obs_channels)`.
