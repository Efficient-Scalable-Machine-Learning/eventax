# Feed Forward Event-driven Neural Network (FFEvNN)

### `class FFEvNN(EvNN)`

A feed‑forward EvNN with layered connectivity.

**Constructor**

```python
FFEvNN(
    layers: list[int],
    in_size: int,
    neuron_model: NeuronModel,
    max_solver_time: float,
    key: jax.random.PRNGKey | None = None,
    init_delays: jnp.ndarray | float | int | None = None,
    dlim: float | None = None,
    buffer_capacity: int = 500,
    solver_stepsize: float = 1e-2,
    solver_checkpoints: int | None = None,
    max_event_steps: int = 100,
    output_no_spike_value: float | None = None,
    root_finder=None,
    stepsize_controller=None,
    solver=None,
    dtype=jnp.float32,
    **neuron_model_kwargs,
)
```

**Behavior**

* Builds a strict feed‑forward mask connecting layer `ℓ` to layer `ℓ+1`.
* Marks the **first** layer as `input_neurons` and the **last** as `output_neurons`.
* Delegates the rest to the base `EvNN` constructor.
