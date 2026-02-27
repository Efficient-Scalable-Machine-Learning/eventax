import jax.numpy as jnp
import equinox as eqx
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .initializations import init_weights_and_bias


def _to_1d(
    x: jnp.ndarray,
    length: int,
    *,
    shared: bool,
    name: str,
) -> jnp.ndarray:
    if shared:
        if x.shape in [(), (1,)]:
            return jnp.reshape(x, (1,))
        raise ValueError(
            f"Shared parameter `{name}` must be scalar or (1,), got {x.shape}."
        )
    else:
        if x.shape in [(), (1,)]:
            return jnp.full((length,), float(x.reshape(())), dtype=x.dtype)
        if x.shape != (length,):
            raise ValueError(
                f"Per-neuron parameter `{name}` must be shape ({length},), got {x.shape}."
            )
        return x


class pIzhikevich(NeuronModel):
    log_a: Float[Array, " a_dim"]
    log_b: Float[Array, " b_dim"]
    c: Float[Array, " c_dim"]
    log_d: Float[Array, " d_dim"]

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
        shared_izhi_params: bool = True,
        wlim: Optional[float] = None,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_weights: bool = True,
        blim: Optional[float] = None,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_bias: bool = True,
        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
        reset_grad_preserve: bool = False,
    ):
        super().__init__(dtype=dtype)

        self.weights, self.ic = init_weights_and_bias(
            key,
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

        a_arr = jnp.asarray(a, dtype=dtype)
        b_arr = jnp.asarray(b, dtype=dtype)
        d_arr = jnp.asarray(d, dtype=dtype)

        if jnp.any(a_arr <= 0):
            raise ValueError("`a` must be strictly positive (> 0).")
        if jnp.any(b_arr <= 0):
            raise ValueError("`b` must be strictly positive (> 0).")
        if jnp.any(d_arr <= 0):
            raise ValueError("`d` must be strictly positive (> 0).")

        a_shaped = _to_1d(a_arr, n_neurons, shared=shared_izhi_params, name="a")
        b_shaped = _to_1d(b_arr, n_neurons, shared=shared_izhi_params, name="b")
        d_shaped = _to_1d(d_arr, n_neurons, shared=shared_izhi_params, name="d")

        self.log_a = jnp.log(a_shaped)
        self.log_b = jnp.log(b_shaped)
        self.log_d = jnp.log(d_shaped)

        c_arr = jnp.asarray(c, dtype=dtype)
        self.c = _to_1d(c_arr, n_neurons, shared=shared_izhi_params, name="c")

        self.thresh = StaticArray(jnp.asarray(v_thresh, dtype=dtype))
        self.tau_syn = StaticArray(jnp.asarray(tau_syn, dtype=dtype))
        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 3"]:
        if self.c.shape == (1,):
            v0 = jnp.full((n_neurons,), float(self.c.reshape(())), dtype=self.ic.dtype)
        else:
            v0 = self.c

        b_param = jnp.exp(self.log_b)
        if b_param.shape == (1,):
            b_full = jnp.full(
                (n_neurons,), float(b_param.reshape(())), dtype=self.ic.dtype
            )
        else:
            b_full = b_param

        u0 = b_full * v0
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

        a = jnp.exp(self.log_a)
        b = jnp.exp(self.log_b)

        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + self.ic + i
        du = a * (b * v - u)
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

        d = jnp.exp(self.log_d)

        if self.reset_grad_preserve:
            delta_v = self.thresh.value - (self.c - self.epsilon)
            v = v - spiked_mask.astype(self.dtype) * delta_v
            u = u + spiked_mask.astype(self.dtype) * d
        else:
            v = jnp.where(spiked_mask, self.c - self.epsilon, v)
            u = jnp.where(spiked_mask, u + d, u)

        return jnp.stack([v, u, i], axis=1)
