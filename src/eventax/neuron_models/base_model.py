import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from typing import Any, Dict
from jaxtyping import Array, Float, Int, Bool


@dataclass(frozen=True, eq=False)
class StaticArray:
    """A wrapper around JAX arrays that should not be optimized.

    Wraps a `jnp.array` in a frozen dataclass to avoid the warning raised by
    `eqx.field(static=True)` on unhashable JAX arrays, while preserving
    hashability via object identity.
    """

    value: Array
    """The underlying JAX array."""

    def __eq__(self, other):
        if not isinstance(other, StaticArray):
            return False
        return self.value is other.value

    def __hash__(self):
        return hash(id(self.value))


class NeuronModel(eqx.Module):
    """Abstract base class for all neuron models.

    Subclasses must implement [`init_state`][eventax.neuron_models.base_model.NeuronModel.init_state],
    [`dynamics`][eventax.neuron_models.base_model.NeuronModel.dynamics],
    [`spike_condition`][eventax.neuron_models.base_model.NeuronModel.spike_condition],
    [`input_spike`][eventax.neuron_models.base_model.NeuronModel.input_spike],
    and [`reset_spiked`][eventax.neuron_models.base_model.NeuronModel.reset_spiked].
    """

    dtype: jnp.dtype = eqx.field(static=True)
    """The JAX dtype used for all internal computations."""

    def __init__(self, dtype: jnp.dtype):
        self.dtype = dtype

    def __call__(self, t, y, args):
        return self.dynamics(t, y, args)

    def init_state(self, n_neurons: int) -> Any:
        """Return the initial state for all neurons in the network."""
        raise NotImplementedError

    def dynamics(
        self,
        t: float,
        y: Any,
        args: Dict[str, Any],
    ) -> Any:
        """Compute the time derivative of the neuron state."""
        raise NotImplementedError

    def spike_condition(
        self,
        t: float,
        y: Any,
        args: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        """Evaluate the spike condition for each neuron."""
        raise NotImplementedError

    def input_spike(
        self,
        y: Any,
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Any:
        """Update state in response to an incoming spike."""
        raise NotImplementedError

    def reset_spiked(
        self,
        y: Any,
        spiked_mask: Bool[Array, "neurons"],
    ) -> Any:
        """Reset the state of neurons that just spiked."""
        raise NotImplementedError

    def observe(
        self,
        y: Any,
    ) -> Float[Array, "neurons obs_channels"]:
        """Extract observable channels from the state."""
        return y
