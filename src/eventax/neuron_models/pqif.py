import jax.numpy as jnp
from typing import Any, Dict, Optional, Union

from jaxtyping import Array, Float, PRNGKeyArray
from .plif import PLIF


class PQIF(PLIF):

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        blim: Optional[float] = None,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_bias: bool = True,
        wlim: Optional[float] = None,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        positive_weights: bool = True,
        init_tmem: Optional[Union[int, float, jnp.ndarray]] = None,
        init_tsyn: Optional[Union[int, float, jnp.ndarray]] = None,
        tmem_low: Optional[float] = None,
        tmem_high: Optional[float] = None,
        tsyn_low: Optional[float] = None,
        tsyn_high: Optional[float] = None,
        shared_time_constants: bool = True,
        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            key=key,
            n_neurons=n_neurons,
            in_size=in_size,
            wmask=wmask,
            init_tmem=init_tmem,
            init_tsyn=init_tsyn,
            tmem_low=tmem_low,
            tsyn_low=tsyn_low,
            tmem_high=tmem_high,
            tsyn_high=tsyn_high,
            shared_time_constants=shared_time_constants,
            blim=blim,
            init_bias=init_bias,
            positive_bias=positive_bias,
            wlim=wlim,
            init_weights=init_weights,
            positive_weights=positive_weights,
            thresh=1.0,
            vreset=-1.0,
            fan_in_mode=fan_in_mode,
            dtype=dtype,
            reset_grad_preserve=False,
        )

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        phi0: Float[Array, "neurons"] = jnp.full((n_neurons,), 0.5, dtype=self.dtype)
        i0: Float[Array, "neurons"] = jnp.zeros((n_neurons,), dtype=self.dtype)
        return jnp.stack([phi0, i0], axis=1)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        phi: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        tmem: Float[Array, "neurons"] = jnp.exp(self.log_tmem)
        tsyn: Float[Array, "neurons"] = jnp.exp(self.log_tsyn)
        c = jnp.cos(jnp.pi * phi)
        s = jnp.sin(jnp.pi * phi)
        term1 = c * (c + (1.0 / jnp.pi) * s)
        term2 = (1.0 / (jnp.pi**2)) * (s**2) * (i + self.ic)
        dphi = (term1 + term2) / tmem
        di = -i / tsyn
        return jnp.stack([dphi, di], axis=1)
