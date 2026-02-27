import equinox as eqx
import jax.numpy as jnp
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .initializations import init_weights_and_bias
from .helpers import clip_with_identity_grad


class EIF(NeuronModel):
    tmem: StaticArray = eqx.field(static=True)
    tsyn: StaticArray = eqx.field(static=True)
    vreset: StaticArray = eqx.field(static=True)
    vT: StaticArray = eqx.field(static=True)
    deltaT: StaticArray = eqx.field(static=True)
    EL: StaticArray = eqx.field(static=True)
    v_peak: StaticArray = eqx.field(static=True)

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
        tsyn: Union[int, float, jnp.ndarray],
        vT: Union[int, float, jnp.ndarray] = 1.0,
        deltaT: Union[int, float, jnp.ndarray] = 0.2,
        EL: Union[int, float, jnp.ndarray] = 0.0,
        vreset: Union[int, float, jnp.ndarray] = 0.0,
        blim: Optional[float] = 0.05,
        bmean: Union[int, float, jnp.ndarray] = 0.025,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        wlim: Optional[float] = 40.0,
        wmean: Union[int, float, jnp.ndarray] = 20.0,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        fan_in_mode: Optional[str] = "linear",
        v_peak: Optional[Union[int, float, jnp.ndarray]] = None,
        spike_time_tol: Optional[Union[int, float, jnp.ndarray]] = 0.001,
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
            wmean=wmean,
            blim=blim,
            bmean=bmean,
            init_bias=init_bias,
            dtype=dtype,
            wmask=wmask,
            fan_in_mode=fan_in_mode,
        )

        self.tmem = StaticArray(jnp.asarray(tmem, dtype=dtype))
        if self.tmem.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`tmem` must be scalar or shape ({n_neurons},); got {self.tmem.value.shape}")

        self.tsyn = StaticArray(jnp.asarray(tsyn, dtype=dtype))
        if self.tsyn.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`tsyn` must be scalar or shape ({n_neurons},); got {self.tsyn.value.shape}")

        self.vreset = StaticArray(jnp.asarray(vreset, dtype=dtype))
        if self.vreset.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`vreset` must be scalar or shape ({n_neurons},); got {self.vreset.value.shape}")

        self.vT = StaticArray(jnp.asarray(vT, dtype=dtype))
        if self.vT.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`vT` must be scalar or shape ({n_neurons},); got {self.vT.value.shape}")

        self.deltaT = StaticArray(jnp.asarray(deltaT, dtype=dtype))
        if self.deltaT.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`deltaT` must be scalar or shape ({n_neurons},); got {self.deltaT.value.shape}")
        if jnp.any(self.deltaT.value <= 0):
            raise ValueError("`deltaT` must be > 0.")

        self.EL = StaticArray(jnp.asarray(EL, dtype=dtype))
        if self.EL.value.shape not in ((), (n_neurons,)):
            raise ValueError(f"`EL` must be scalar or shape ({n_neurons},); got {self.EL.value.shape}")

        if (v_peak is not None) and (spike_time_tol is not None):
            raise ValueError("Specify either `v_peak` or `spike_time_tol`, not both.")

        if spike_time_tol is not None:
            tol = jnp.asarray(spike_time_tol, dtype=dtype)
            if tol.shape not in ((), (n_neurons,)):
                raise ValueError(
                    f"`spike_time_tol` must be scalar or shape ({n_neurons},); got {tol.shape}"
                )
            if jnp.any(tol <= 0):
                raise ValueError("`spike_time_tol` must be > 0.")

            v_peak_val = self.vT.value - self.deltaT.value * jnp.log(
                tol / self.tmem.value
            )
            self.v_peak = StaticArray(v_peak_val.astype(dtype))
        else:
            if v_peak is None:
                v_peak_arr = self.vT.value + 5.0 * self.deltaT.value
            else:
                v_peak_arr = jnp.asarray(v_peak, dtype=dtype)

            if v_peak_arr.shape not in ((), (n_neurons,)):
                raise ValueError(
                    f"`v_peak` must be scalar or shape ({n_neurons},); got {v_peak_arr.shape}"
                )
            self.v_peak = StaticArray(v_peak_arr)

        self.epsilon = jnp.finfo(dtype).eps.item()
        self.reset_grad_preserve = reset_grad_preserve

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        V0 = jnp.full((n_neurons,), self.EL.value if self.EL.value.shape == () else 0.0, dtype=self.dtype)
        if self.EL.value.shape == (n_neurons,):
            V0 = self.EL.value
        i0: Float[Array, "neurons"] = jnp.zeros((n_neurons,), dtype=self.dtype)
        return jnp.stack([V0, i0], axis=1)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        V: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        I_tot = i + self.ic

        u = (V - self.vT.value) / self.deltaT.value
        exp_term = jnp.exp(u)

        dV = (self.EL.value - V + self.deltaT.value * exp_term + I_tot) / self.tmem.value
        di = -i / self.tsyn.value

        return jnp.stack([dV, di], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        V = y[:, 0]
        return V - self.v_peak.value

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
        V, i = y[:, 0], y[:, 1]
        delta_v = self.v_peak.value - self.vreset.value

        if self.reset_grad_preserve:
            V = V - spike_mask.astype(self.dtype) * delta_v
        else:
            V = jnp.where(spike_mask, self.vreset.value, V)

        V = clip_with_identity_grad(V, self.v_peak.value - self.epsilon)
        return jnp.stack([V, i], axis=1)

    def observe(
        self,
        y: Float[Array, "neurons 2"],
    ) -> Float[Array, "neurons 2"]:
        V: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        return jnp.stack([V, i], axis=1)
