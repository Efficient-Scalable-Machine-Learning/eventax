import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .initializations import init_weights_and_bias
from .helpers import clip_with_identity_grad


def _to_1d(x: jnp.ndarray, length: int, *, shared: bool) -> jnp.ndarray:
    if shared:
        if x.shape in [(), (1,)]:
            return jnp.reshape(x, (1,))
        raise ValueError(f"Shared parameter must be scalar or (1,), got {x.shape}.")
    else:
        if x.shape in [(), (1,)]:
            return jnp.full((length,), float(x.reshape(())), dtype=x.dtype)
        if x.shape != (length,):
            raise ValueError(f"Per-neuron parameter must be shape ({length},), got {x.shape}.")
        return x


def _make_positive_param(
    *,
    init: Optional[Union[int, float, jnp.ndarray]],
    low: Optional[float],
    high: Optional[float],
    key: PRNGKeyArray,
    n_neurons: int,
    shared: bool,
    dtype,
) -> jnp.ndarray:
    if init is not None:
        arr = jnp.asarray(init, dtype=dtype)
        if jnp.any(arr <= 0):
            raise ValueError("Time constants must be > 0.")
        return _to_1d(arr, n_neurons, shared=shared)

    if low is None or high is None or not (low > 0 and high > low):
        raise ValueError("Provide 0 < low < high for randomized time constants.")
    if shared:
        val = jax.random.uniform(key, (), minval=low, maxval=high, dtype=dtype)
        return jnp.reshape(val, (1,))
    else:
        return jax.random.uniform(key, (n_neurons,), minval=low, maxval=high, dtype=dtype)


class PLIF(NeuronModel):
    thresh: StaticArray = eqx.field(static=True)
    vreset: StaticArray = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    reset_grad_preserve: bool = eqx.field(static=True)

    weights: Float[Array, "in_plus_neurons neurons"]
    ic: Float[Array, "neurons"]
    log_tmem: Float[Array, "neurons"]
    log_tsyn: Float[Array, "neurons"]

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],

        thresh: Union[int, float, jnp.ndarray],
        vreset: Union[int, float, jnp.ndarray] = 0.0,

        wlim: Optional[float] = None,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_weights: bool = True,

        blim: Optional[float] = None,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_bias: bool = True,

        init_tmem: Optional[Union[int, float, jnp.ndarray]] = None,
        init_tsyn: Optional[Union[int, float, jnp.ndarray]] = None,
        tmem_low: Optional[float] = None,
        tmem_high: Optional[float] = None,
        tsyn_low: Optional[float] = None,
        tsyn_high: Optional[float] = None,
        shared_time_constants: bool = True,

        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
        reset_grad_preserve: bool = True,
    ):
        super().__init__(dtype=dtype)

        need_w = init_weights is None
        need_b = init_bias is None
        need_tm = init_tmem is None
        need_ts = init_tsyn is None

        if (need_w or need_b or need_tm or need_ts) and key is None:
            raise ValueError("Provide `key` when any of weights/bias/tmem/tsyn are randomized.")

        if key is not None:
            wbkey, tmkey, tskey = jax.random.split(key, 3)
        else:
            wbkey = tmkey = tskey = jax.random.PRNGKey(0)

        self.weights, self.ic = init_weights_and_bias(
            wbkey,
            n_neurons=n_neurons,
            in_size=in_size,
            wlim=wlim,
            init_weights=init_weights,
            positive_weights=positive_weights,
            blim=blim,
            init_bias=init_bias,
            positive_bias=positive_bias,
            dtype=dtype,
            wmask=wmask,
            fan_in_mode=fan_in_mode,
        )

        tmem = _make_positive_param(
            init=init_tmem, low=tmem_low, high=tmem_high, key=tmkey,
            n_neurons=n_neurons, shared=shared_time_constants, dtype=dtype
        )
        tsyn = _make_positive_param(
            init=init_tsyn, low=tsyn_low, high=tsyn_high, key=tskey,
            n_neurons=n_neurons, shared=shared_time_constants, dtype=dtype
        )
        self.log_tmem = jnp.log(tmem)
        self.log_tsyn = jnp.log(tsyn)

        self.thresh = StaticArray(jnp.asarray(thresh, dtype=dtype))
        if self.thresh.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`thresh` must be scalar or shape ({n_neurons},); got {self.thresh.value.shape}")

        self.vreset = StaticArray(jnp.asarray(vreset, dtype=dtype))
        if self.vreset.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`vreset` must be scalar or shape ({n_neurons},); got {self.vreset.value.shape}")

        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        """State layout: [:, 0] = v (voltage), [:, 1] = i (synaptic current)."""
        return jnp.zeros((n_neurons, 2), dtype=self.dtype)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        v: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        tmem: Float[Array, "neurons"] = jnp.exp(self.log_tmem)
        tsyn: Float[Array, "neurons"] = jnp.exp(self.log_tsyn)
        dv_dt = (-v + i + self.ic) / tmem
        di_dt = -i / tsyn
        return jnp.stack([dv_dt, di_dt], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        return y[:, 0] - self.thresh.value

    def input_spike(
        self,
        y: Float[Array, "neurons 2"],
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Float[Array, "neurons 2"]:
        delta_i = self.weights[from_idx, to_idx] * valid_mask
        return y.at[to_idx, 1].add(delta_i)

    def reset_spiked(
        self,
        y: Float[Array, "neurons 2"],
        spike_mask: Bool[Array, "neurons"],
    ) -> Float[Array, "neurons 2"]:
        v, i = y[:, 0], y[:, 1]
        delta_v = self.thresh.value - self.vreset.value

        if self.reset_grad_preserve:
            v = v - spike_mask.astype(self.dtype) * delta_v
        else:
            v = jnp.where(spike_mask, self.vreset.value, v)

        v = clip_with_identity_grad(v, self.thresh.value - self.epsilon)
        return jnp.stack([v, i], axis=1)
