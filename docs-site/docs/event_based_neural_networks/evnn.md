# Event-driven Neural Network (EvNN)

Event-driven neural network core built with **JAX**, **Equinox**, and **Diffrax**.

---

## High-level overview

`EvNN` simulates a network of spiking neurons using an event-based formulation:

* An ODE solver integrates the neuron state from the time of the last event to the next event.
* Event times are stored in a sorted buffer.
* At every iteration, we integrate the neuron states from the first event in the buffer to either the next event or until one of the neurons causes a new event.
* When a neuron causes an event, its state is reset and the new events are enqueued.

[`FFEvNN`][eventax.evnn.FFEvNN] is a convenience subclass for creating feed-forward networks from a list of layer sizes, automatically wiring input and output neuron masks.

The network delegates all neuron-specific behaviour to a pluggable [`NeuronModel`][eventax.neuron_models.NeuronModel], which must expose a consistent interface (see [NeuronModel](../neuron_models/neuron_model.md)).

::: eventax.evnn.EvNN
    options:
      members: false

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `neuron_model` | Class (not instance) implementing the [`NeuronModel`][eventax.neuron_models.NeuronModel] interface. Constructed internally using `key`, `n_neurons`, `in_size`, `dtype`, and any extra `**neuron_model_kwargs`. | — |
| `n_neurons` | Number of neurons in the network. | — |
| `max_solver_time` | Hard stop time for the simulation; spikes beyond this time are discarded. | — |
| `in_size` | Number of input slots. Each slot may carry multiple input spikes. | — |
| `key` | Random key for initialising the neuron model. | `None` |
| `t0` | Simulation start time. | `0.0` |
| `wmask` | Connectivity mask from all senders (neurons + inputs) to neurons, shape $(N+K) \times N$. If omitted, defaults to all-to-all among neurons plus inputs → selected first layer (`input_neurons`). | `None` |
| `output_neurons` | Boolean mask $(N,)$ indicating which neurons are output neurons. | `None` (all) |
| `input_neurons` | Boolean mask $(N,)$ selecting neurons that receive external input spikes. | `None` |
| `dtype` | Numeric type for event times and dynamics. | `jnp.float32` |
| `**neuron_model_kwargs` | Additional keyword arguments passed to the neuron model constructor. | — |

### Solver configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `solver` | ODE solver (any `diffrax.AbstractSolver`). | `diffrax.Euler()` |
| `stepsize_controller` | Step-size control strategy. | `diffrax.ConstantStepSize()` |
| `solver_stepsize` | Initial step size for the ODE solver. | `1e-3` |
| `root_finder` | Root finder used in `diffrax.Event` to locate spike times. | `optimistix.Newton(1e-2, 1e-2, rms_norm)` |
| `adjoint` | Adjoint method for differentiating through `diffeqsolve`. | `diffrax.RecursiveCheckpointAdjoint()` |

### Simulation bounds

| Parameter | Description | Default |
|-----------|-------------|---------|
| `buffer_capacity` | Capacity of the internal spike buffer. | `1` |
| `max_event_steps` | Upper bound on the number of event-loop iterations. | `1000` |
| `output_no_spike_value` | Fill value for "no spike yet" in output arrays. | `jnp.inf` |

---

## Methods

### `init_state`

::: eventax.evnn.EvNN.init_state
    options:
      show_root_heading: false

Returns the initial neuron state from the [`NeuronModel`][eventax.neuron_models.NeuronModel], cast to the network's `dtype`.

---

### `init_buffer`

::: eventax.evnn.EvNN.init_buffer
    options:
      show_root_heading: false

Builds a spike buffer for the simulation. `in_spike_times` must have shape `(in_size, K)` where `K` is the maximum number of spikes per input slot. Use `jnp.inf` for unused slots.

If `comp_times` is provided, pseudospike events are inserted at those times so that [`state_at_t`][eventax.evnn.EvNN.state_at_t] can record the neuron state.

---

### `__call__`

::: eventax.evnn.EvNN.__call__
    options:
      show_root_heading: false

Processes a single event-integration window:

1. Pops the next event from the buffer.
2. Applies any incoming spike to the neuron state via [`input_spike`][eventax.neuron_models.NeuronModel.input_spike].
3. Integrates the ODE from the current event time to the next event (or until a neuron spikes).
4. If a neuron spiked, resets its state via [`reset_spiked`][eventax.neuron_models.NeuronModel.reset_spiked] and enqueues the new spike event.

**Returns** a tuple `(t_event, spike_mask, new_state, new_buffer)`.

---

### `ttfs`

::: eventax.evnn.EvNN.ttfs
    options:
      show_root_heading: false

Computes **time-to-first-spike** for each output neuron. Iterates the event loop until all outputs have spiked or `max_solver_time` is reached. Returns an array of shape `(n_outputs,)` filled with `output_no_spike_value` for neurons that did not spike.

---

### `spikes_until_t`

::: eventax.evnn.EvNN.spikes_until_t
    options:
      show_root_heading: false

Records up to `max_spikes` spike times per output neuron up to `final_time`. Returns an array of shape `(n_outputs, max_spikes)` filled with `output_no_spike_value` where unused.

---

### `state_at_t`

::: eventax.evnn.EvNN.state_at_t
    options:
      show_root_heading: false

Returns the observable state of output neurons at the specified computation times. Uses pseudospike events to halt integration at the requested times. Returns an array of shape `(n_outputs, n_times, obs_channels)`.

---

### `record`

::: eventax.evnn.EvNN.record
    options:
      show_root_heading: false

Runs the full event loop up to `max_event_steps`, recording spike times, neuron IDs, and buffer sizes at each step. Useful for debugging and visualisation.

**Returns** a tuple `(spike_times, spike_ids, buffer_sizes)`, each of shape `(max_event_steps,)`.

---

### `get_wmask`

::: eventax.evnn.EvNN.get_wmask
    options:
      show_root_heading: false

Reconstructs a dense $\{0, 1\}$ connectivity mask of shape $(N+K) \times N$ from the internal `syn_conn` representation.
