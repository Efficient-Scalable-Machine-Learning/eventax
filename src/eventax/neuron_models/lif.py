import equinox as eqx
import jax.numpy as jnp
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .initializations import init_weights_and_bias
from .helpers import clip_with_identity_grad


class LIF(NeuronModel):
    """Leaky integrate-and-fire neuron with current-based synapse.

    Two state channels: membrane voltage $v$ and synaptic current $i$.
    """

    thresh: StaticArray = eqx.field(static=True)
    """Spike threshold $v_\\mathrm{th}$. Scalar or per-neuron."""

    tmem: StaticArray = eqx.field(static=True)
    """Membrane time constant $\\tau_\\mathrm{mem}$. Scalar or per-neuron."""

    tsyn: StaticArray = eqx.field(static=True)
    """Synaptic time constant $\\tau_\\mathrm{syn}$. Scalar or per-neuron."""

    vreset: StaticArray = eqx.field(static=True)
    """Reset voltage $v_\\mathrm{reset}$. Scalar or per-neuron."""

    epsilon: float = eqx.field(static=True)
    """Machine epsilon for the chosen dtype."""

    reset_grad_preserve: bool = eqx.field(static=True)
    """If `True`, reset uses subtraction to preserve gradients."""

    weights: Float[Array, "in_plus_neurons neurons"]
    """Connection weight matrix."""

    ic: Float[Array, "neurons"]
    """Learnable bias current $i_c$."""

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        thresh: Union[int, float, jnp.ndarray],
        tsyn: Union[int, float, jnp.ndarray],
        tmem: Union[int, float, jnp.ndarray],
        vreset: Union[int, float, jnp.ndarray] = 0,
        blim: Optional[float] = None,
        bmean: Union[int, float, jnp.ndarray] = 0.0,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        wlim: Optional[float] = None,
        wmean: Union[int, float, jnp.ndarray] = 0.0,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
        reset_grad_preserve: bool = True,
    ):
        super().__init__(dtype=dtype)

        self.weights, self.ic = init_weights_and_bias(
            key,
            n_neurons=n_neurons,
            in_size=in_size,
            wlim=wlim,
            wmean=wmean,
            init_weights=init_weights,
            blim=blim,
            bmean=bmean,
            init_bias=init_bias,
            dtype=dtype,
            wmask=wmask,
            fan_in_mode=fan_in_mode,
        )

        self.thresh = StaticArray(jnp.asarray(thresh, dtype=dtype))
        if self.thresh.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`thresh` must be scalar or shape ({n_neurons},); got {self.thresh.value.shape}")

        self.tmem = StaticArray(jnp.asarray(tmem, dtype=dtype))
        if self.tmem.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`tmem` must be scalar or shape ({n_neurons},); got {self.tmem.value.shape}")

        self.tsyn = StaticArray(jnp.asarray(tsyn, dtype=dtype))
        if self.tsyn.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`tsyn` must be scalar or shape ({n_neurons},); got {self.tsyn.value.shape}")

        self.vreset = StaticArray(jnp.asarray(vreset, dtype=dtype))
        if self.vreset.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`vreset` must be scalar or shape ({n_neurons},); got {self.vreset.value.shape}")

        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        """Return zero-initialised state of shape `(n_neurons, 2)`."""
        return jnp.zeros((n_neurons, 2), dtype=self.dtype)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        """Compute the LIF ODE derivatives for voltage and synaptic current."""
        v = y[:, 0]
        i = y[:, 1]
        dv_dt = (-v + i + self.ic) / self.tmem.value
        di_dt = -i / self.tsyn.value
        return jnp.stack([dv_dt, di_dt], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        """Return `v - thresh`; sign change triggers a spike."""
        return y[:, 0] - self.thresh.value

    def input_spike(
        self,
        y: Float[Array, "neurons 2"],
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Float[Array, "neurons 2"]:
        """Add connection weights to the synaptic current of target neurons."""
        delta_i = self.weights[from_idx, to_idx] * valid_mask
        return y.at[to_idx, 1].add(delta_i)

    def reset_spiked(
        self,
        y: Float[Array, "neurons 2"],
        spike_mask: Bool[Array, "neurons"],
    ) -> Float[Array, "neurons 2"]:
        """Reset voltage of spiked neurons and clip to prevent re-triggering."""
        v, i = y[:, 0], y[:, 1]
        delta_v = self.thresh.value - self.vreset.value

        if self.reset_grad_preserve:
            v = v - spike_mask.astype(self.dtype) * delta_v
        else:
            v = jnp.where(spike_mask, self.vreset.value, v)

        v = clip_with_identity_grad(v, self.thresh.value - self.epsilon)
        return jnp.stack([v, i], axis=1)
