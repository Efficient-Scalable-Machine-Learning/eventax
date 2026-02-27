import jax.numpy as jnp
import equinox as eqx
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .initializations import init_weights_and_bias
from .helpers import clip_with_identity_grad


class Izhikevich(NeuronModel):
    a: StaticArray = eqx.field(static=True)
    b: StaticArray = eqx.field(static=True)
    c: StaticArray = eqx.field(static=True)
    d: StaticArray = eqx.field(static=True)
    thresh: StaticArray = eqx.field(static=True)
    tau_syn: StaticArray = eqx.field(static=True)
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
        *,
        a: Union[int, float, jnp.ndarray] = 0.02,
        b: Union[int, float, jnp.ndarray] = 0.2,
        c: Union[int, float, jnp.ndarray] = -51.0,
        d: Union[int, float, jnp.ndarray] = 2.0,
        v_thresh: Union[int, float, jnp.ndarray] = 30.0,
        tau_syn: Union[int, float, jnp.ndarray] = 5.0,
        wlim: Optional[float] = None,
        wmean: Union[int, float, jnp.ndarray] = 0.0,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        blim: Optional[float] = None,
        bmean: Union[int, float, jnp.ndarray] = 0.0,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
        reset_grad_preserve: bool = True,
    ):
        super().__init__(dtype=dtype)

        self.weights, self.ic = init_weights_and_bias(
            key,
            n_neurons=n_neurons,
            in_size=in_size,
            wmask=wmask,
            wlim=wlim,
            wmean=wmean,
            init_weights=init_weights,
            blim=blim,
            bmean=bmean,
            init_bias=init_bias,
            dtype=dtype,
            fan_in_mode=fan_in_mode,
        )

        self.a = StaticArray(jnp.asarray(a, dtype=dtype))
        self.b = StaticArray(jnp.asarray(b, dtype=dtype))
        self.c = StaticArray(jnp.asarray(c, dtype=dtype))
        self.d = StaticArray(jnp.asarray(d, dtype=dtype))
        self.thresh = StaticArray(jnp.asarray(v_thresh, dtype=dtype))
        self.tau_syn = StaticArray(jnp.asarray(tau_syn, dtype=dtype))
        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 3"]:
        v0 = jnp.full((n_neurons,), self.c.value, dtype=self.ic.dtype)
        u0 = self.b.value * v0
        i0 = jnp.zeros((n_neurons,), dtype=self.ic.dtype)
        return jnp.stack([v0, u0, i0], axis=1)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 3"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 3"]:
        v = y[:, 0]
        u = y[:, 1]
        i = y[:, 2]

        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + self.ic + i
        du = self.a.value * (self.b.value * v - u)
        di = -i / self.tau_syn.value

        return jnp.stack([dv, du, di], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 3"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        return y[:, 0] - self.thresh.value

    def input_spike(
        self,
        y: Float[Array, "neurons 3"],
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Float[Array, "neurons 3"]:
        v = y[:, 0]
        u = y[:, 1]
        i = y[:, 2]

        delta_i = self.weights[from_idx, to_idx] * valid_mask
        i = i.at[to_idx].add(delta_i)

        return jnp.stack([v, u, i], axis=1)

    def reset_spiked(
        self,
        y: Float[Array, "neurons 3"],
        spiked_mask: Bool[Array, "neurons"],
    ) -> Float[Array, "neurons 3"]:
        v = y[:, 0]
        u = y[:, 1]
        i = y[:, 2]

        if self.reset_grad_preserve:
            delta_v = self.thresh.value - (self.c.value - self.epsilon)
            v = v - spiked_mask.astype(self.dtype) * delta_v
            u = u + spiked_mask.astype(self.dtype) * self.d.value
        else:
            v = jnp.where(spiked_mask, self.c.value - self.epsilon, v)
            u = jnp.where(spiked_mask, u + self.d.value, u)

        v = clip_with_identity_grad(v, self.thresh.value - self.epsilon)
        return jnp.stack([v, u, i], axis=1)
