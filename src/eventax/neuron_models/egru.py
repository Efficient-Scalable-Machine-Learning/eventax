import jax
import equinox as eqx
import jax.numpy as jnp
from typing import Union, Dict, Any
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from .base_model import NeuronModel, StaticArray
from .helpers import clip_with_identity_grad


class EGRU(NeuronModel):
    """Event-based gated recurrent unit with continuous-time dynamics.

    Four state channels: cell state $c$ and three pre-activations $a_u, a_r, a_z$.
    """

    W_u: Float[Array, "in_plus_neurons neurons"]
    """Weight matrix for the update gate."""

    W_r: Float[Array, "in_plus_neurons neurons"]
    """Weight matrix for the reset gate."""

    W_z: Float[Array, "in_plus_neurons neurons"]
    """Weight matrix for the modulation gate."""

    b_u: Float[Array, "neurons"]
    """Bias attractor for the update pre-activation $a_u$."""

    b_r: Float[Array, "neurons"]
    """Bias attractor for the reset pre-activation $a_r$."""

    b_z: Float[Array, "neurons"]
    """Bias attractor for the modulation pre-activation $a_z$."""

    tsyn: StaticArray = eqx.field(static=True)
    """Synaptic time constant $\\tau_\\mathrm{syn}$."""

    tmem: StaticArray = eqx.field(static=True)
    """Membrane time constant $\\tau_\\mathrm{mem}$."""

    thresh: StaticArray = eqx.field(static=True)
    """Spike threshold $\\vartheta$ applied to cell state $c$."""

    epsilon: float = eqx.field(static=True)
    """Machine epsilon for the chosen dtype."""

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        tsyn: Union[int, float, jnp.ndarray] = 5.0,
        tmem: Union[int, float, jnp.ndarray] = 20.0,
        thresh: Union[int, float, jnp.ndarray] = 0.5,
        bias_u_mean: float = -2.0,
        bias_r_mean: float = 1.0,
        bias_z_mean: float = 0.0,
        bias_scale: float = 0.1,
        weight_scale: float = 1.0,
        weight_mean: float = 0.0,
        dtype=jnp.float32,
    ):
        super().__init__(dtype=dtype)

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        self.b_u = bias_scale * jax.random.normal(k1, (n_neurons,), dtype=dtype) + bias_u_mean
        self.b_r = bias_scale * jax.random.normal(k2, (n_neurons,), dtype=dtype) + bias_r_mean
        self.b_z = bias_scale * jax.random.normal(k3, (n_neurons,), dtype=dtype) + bias_z_mean

        self.W_u = weight_scale * jax.random.normal(k4, (n_neurons + in_size, n_neurons), dtype=dtype) + weight_mean
        self.W_r = weight_scale * jax.random.normal(k5, (n_neurons + in_size, n_neurons), dtype=dtype) + weight_mean
        self.W_z = weight_scale * jax.random.normal(k6, (n_neurons + in_size, n_neurons), dtype=dtype) + weight_mean

        self.tsyn = StaticArray(jnp.asarray(tsyn, dtype=dtype))
        self.tmem = StaticArray(jnp.asarray(tmem, dtype=dtype))
        self.thresh = StaticArray(jnp.asarray(thresh, dtype=dtype))
        self.epsilon = jnp.finfo(dtype).eps.item()

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 4"]:
        """Return zero-initialised state of shape `(n_neurons, 4)`."""
        return jnp.zeros((n_neurons, 4), dtype=self.dtype)

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 4"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 4"]:
        """Compute the EGRU ODE derivatives for cell state and pre-activations."""
        c = y[:, 0]
        a_u = y[:, 1]
        a_r = y[:, 2]
        a_z = y[:, 3]

        da_u = (-a_u + self.b_u) / self.tsyn.value
        da_r = (-a_r + self.b_r) / self.tsyn.value
        da_z = (-a_z + self.b_z) / self.tsyn.value

        u = jax.nn.sigmoid(a_u)
        z = jnp.tanh(a_z)
        dc = (u * (z - c)) / self.tmem.value

        return jnp.stack([dc, da_u, da_r, da_z], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 4"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        """Return `c - thresh`; sign change triggers a spike."""
        return y[:, 0] - self.thresh.value

    def input_spike(
        self,
        y: Float[Array, "neurons 4"],
        from_idx: Union[int, Int[Array, ""]],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Float[Array, "neurons 4"]:
        """Add gated weight contributions to the pre-activation channels of target neurons."""
        res = jax.nn.sigmoid(y[to_idx, 2])
        du = self.W_u[from_idx, to_idx] * valid_mask
        dr = self.W_r[from_idx, to_idx] * valid_mask
        dz = self.W_z[from_idx, to_idx] * res * valid_mask

        y = y.at[to_idx, 1].add(du)
        y = y.at[to_idx, 2].add(dr)
        y = y.at[to_idx, 3].add(dz)
        return y

    def reset_spiked(
        self,
        y: Float[Array, "neurons 4"],
        spiked_mask: Bool[Array, "neurons"],
    ) -> Float[Array, "neurons 4"]:
        """Reset cell state of spiked neurons via subtraction and clip to prevent re-triggering."""
        c, a_u, a_r, a_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        c = c - spiked_mask.astype(self.dtype) * self.thresh.value
        c = clip_with_identity_grad(c, self.thresh.value - self.epsilon)
        return jnp.stack([c, a_u, a_r, a_z], axis=1)
