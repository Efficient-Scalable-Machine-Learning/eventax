import jax
import equinox as eqx
import jax.numpy as jnp
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .helpers import clip_with_identity_grad


class ALIF(NeuronModel):
    tmem: StaticArray = eqx.field(static=True)
    ta: StaticArray = eqx.field(static=True)
    b0: StaticArray = eqx.field(static=True)
    rm: StaticArray = eqx.field(static=True)
    beta: StaticArray = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    reset_grad_preserve: bool = eqx.field(static=True)

    weights: Float[Array, "in_plus_neurons neurons"]
    ic: Float[Array, "neurons"]

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        tmem: Union[int, float, jnp.ndarray],
        ta: Union[int, float, jnp.ndarray],
        b0: Union[int, float, jnp.ndarray],
        beta: Union[int, float, jnp.ndarray],
        rm: Union[int, float, jnp.ndarray] = 1.0,
        blim: Optional[float] = None,
        init_bias: Optional[Union[jnp.ndarray, int, float]] = None,
        positive_bias: bool = True,
        wlim: Optional[float] = None,
        init_weights: Optional[Union[jnp.ndarray, int, float]] = None,
        positive_weights: bool = True,
        dtype=jnp.float32,
        reset_grad_preserve: bool = True,
    ):
        super().__init__(dtype=dtype)

        if init_weights is None or init_bias is None:
            if key is None:
                raise ValueError(
                    "Must set key, because one of weights, bias is not set and will be generated randomly"
                )
            wkey, bkey = jax.random.split(key, 2)

        if init_weights is None:
            if wlim is None:
                raise ValueError("If init_weights is None, wlim must be set")
            if positive_weights:
                self.weights = jax.random.uniform(
                    wkey, (n_neurons + in_size, n_neurons), minval=0.0, maxval=wlim, dtype=dtype
                )
            else:
                self.weights = jax.random.uniform(
                    wkey, (n_neurons + in_size, n_neurons), minval=-wlim / 2, maxval=wlim / 2, dtype=dtype
                )
        elif isinstance(init_weights, (int, float)):
            self.weights = jnp.full((n_neurons + in_size, n_neurons), init_weights, dtype=dtype)
        else:
            self.weights = jnp.asarray(init_weights, dtype=dtype)

        if init_bias is None:
            if blim is None:
                raise ValueError("if init_bias is None, blim must be set")
            if positive_bias:
                self.ic = jax.random.uniform(bkey, (n_neurons,), minval=0.0, maxval=blim, dtype=dtype)
            else:
                self.ic = jax.random.uniform(bkey, (n_neurons,), minval=-blim / 2, maxval=blim / 2, dtype=dtype)
        elif isinstance(init_bias, (int, float)):
            self.ic = jnp.full((n_neurons,), init_bias, dtype=dtype)
        else:
            self.ic = jnp.asarray(init_bias, dtype=dtype)

        self.tmem = StaticArray(jnp.asarray(tmem, dtype=dtype))
        if self.tmem.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`tmem` must be scalar or shape ({n_neurons},); got {self.tmem.value.shape}")

        self.rm = StaticArray(jnp.asarray(rm, dtype=dtype))
        if self.rm.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`rm` must be scalar or shape ({n_neurons},); got {self.rm.value.shape}")

        self.ta = StaticArray(jnp.asarray(ta, dtype=dtype))
        if self.ta.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`ta` must be scalar or shape ({n_nerons},); got {self.ta.value.shape}")

        self.b0 = StaticArray(jnp.asarray(b0, dtype=dtype))
        if self.b0.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`b0` must be scalar or shape ({n_neurons},); got {self.b0.value.shape}")

        self.beta = StaticArray(jnp.asarray(beta, dtype=dtype))
        if self.beta.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`beta` must be scalar or shape ({n_neurons},); got {self.beta.value.shape}")

        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        v0 = jnp.zeros((n_neurons,), dtype=self.dtype)
        b0 = jnp.broadcast_to(self.b0.value, (n_neurons,))
        return jnp.stack([v0, b0], axis=1)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        v: Float[Array, "neurons"] = y[:, 0]
        b: Float[Array, "neurons"] = y[:, 1]
        dv_dt = (-v + self.ic) / self.tmem.value
        db_dt = (-b + self.b0.value) / self.ta.value
        return jnp.stack([dv_dt, db_dt], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        v: Float[Array, "neurons"] = y[:, 0]
        b: Float[Array, "neurons"] = y[:, 1]
        return v - b

    def input_spike(
        self,
        y: Float[Array, "neurons 2"],
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Float[Array, "neurons 2"]:
        delta_v = self.rm.value * self.weights[from_idx, to_idx] * valid_mask
        return y.at[to_idx, 0].add(delta_v)

    def reset_spiked(
        self,
        y: Float[Array, "neurons 2"],
        spike_mask: Bool[Array, "neurons"],
    ) -> Float[Array, "neurons 2"]:
        v, b = y[:, 0], y[:, 1]
        b_inc = self.beta.value / self.ta.value

        if self.reset_grad_preserve:
            mask_f = spike_mask.astype(self.dtype)
            v = v - mask_f * b
            b = b + mask_f * b_inc
        else:
            v = jnp.where(spike_mask, 0.0, v)
            b = jnp.where(spike_mask, b + b_inc, b)

        v = clip_with_identity_grad(v, b - self.epsilon)
        return jnp.stack([v, b], axis=1)
