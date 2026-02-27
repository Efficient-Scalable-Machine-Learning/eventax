import jax.numpy as jnp
from typing import Any, Dict, Optional, Union
from jaxtyping import Array, Float, PRNGKeyArray
from .lif import LIF


class QIF(LIF):
    """Quadratic integrate-and-fire neuron in phase representation.

    Extends [`LIF`][eventax.neuron_models.LIF] with quadratic voltage dynamics
    reformulated via the phase variable $\\varphi =
    \\frac{1}{\\pi}\\arctan\\!\\left(\\frac{V}{\\pi}\\right) + \\frac{1}{2}$.
    """

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        tmem: Union[int, float, jnp.ndarray],
        tsyn: Union[int, float, jnp.ndarray],
        blim: Optional[float] = None,
        bmean: Union[int, float, jnp.ndarray] = 0.0,
        init_bias: Optional[Union[int, float, jnp.ndarray]] = None,
        wlim: Optional[float] = None,
        wmean: Union[int, float, jnp.ndarray] = 0.0,
        init_weights: Optional[Union[int, float, jnp.ndarray]] = None,
        fan_in_mode: Optional[str] = None,
        dtype=jnp.float32,
    ):
        super().__init__(
            key=key,
            n_neurons=n_neurons,
            in_size=in_size,
            wmask=wmask,
            tmem=tmem,
            tsyn=tsyn,
            blim=blim,
            bmean=bmean,
            init_bias=init_bias,
            wlim=wlim,
            wmean=wmean,
            init_weights=init_weights,
            thresh=1.0,
            vreset=0.0,
            fan_in_mode=fan_in_mode,
            dtype=dtype,
            reset_grad_preserve=False,
        )

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        """Return initial state with $\\varphi = 0.5$ and $I = 0$."""
        phi0: Float[Array, "neurons"] = jnp.full((n_neurons,), 0.5, dtype=self.dtype)
        i0: Float[Array, "neurons"] = jnp.zeros((n_neurons,), dtype=self.dtype)
        return jnp.stack([phi0, i0], axis=1)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        """Compute the QIF phase-space ODE derivatives."""
        phi: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        c = jnp.cos(jnp.pi * phi)
        s = jnp.sin(jnp.pi * phi)
        term1 = c * (c + (1.0 / jnp.pi) * s)
        term2 = (1.0 / (jnp.pi**2)) * (s**2) * (i + self.ic)
        dphi = (term1 + term2) / self.tmem.value
        di = -i / self.tsyn.value
        return jnp.stack([dphi, di], axis=1)

    def observe(
        self,
        y: Float[Array, "neurons 2"],
    ) -> Float[Array, "neurons 2"]:
        """Map phase $\\varphi$ back to voltage via $V = \\pi\\tan\\!\\bigl(\\pi(\\varphi - \\tfrac{1}{2})\\bigr)$."""
        phi: Float[Array, "neurons"] = y[:, 0]
        i: Float[Array, "neurons"] = y[:, 1]
        v: Float[Array, "neurons"] = jnp.pi * jnp.tan(jnp.pi * (phi - 0.5))
        return jnp.stack([v, i], axis=1)
