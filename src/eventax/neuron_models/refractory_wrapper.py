import equinox as eqx
import jax.numpy as jnp
from typing import Any, Dict, Optional, Type, TypeVar

from jaxtyping import Array, Float, Int, Bool
from .base_model import NeuronModel, StaticArray

TNeuron = TypeVar("TNeuron", bound=NeuronModel)


def Refractory(NeuronCls: Type[TNeuron]) -> Type[NeuronModel]:

    class RefractoryNeuron(NeuronModel):
        neuron: NeuronModel
        t_refrac: StaticArray = eqx.field(static=True)
        block_input_during_refrac: bool = eqx.field(static=True)
        refrac_mask: Optional[StaticArray] = eqx.field(static=True)

        def __init__(
            self,
            *args,
            t_refrac: float | jnp.ndarray = 0.0,
            block_input_during_refrac: bool = True,
            refrac_mask: Optional[Array] = None,
            **kwargs,
        ):
            inner = NeuronCls(*args, **kwargs)
            super().__init__(dtype=inner.dtype)
            self.neuron = inner
            tr = jnp.asarray(t_refrac, dtype=inner.dtype)
            self.t_refrac = StaticArray(tr)
            self.block_input_during_refrac = bool(block_input_during_refrac)
            if refrac_mask is None:
                self.refrac_mask = None
            else:
                rm = jnp.asarray(refrac_mask, dtype=inner.dtype)
                if rm.ndim > 1:
                    raise ValueError("`refrac_mask` must be scalar or 1D.")
                self.refrac_mask = StaticArray(rm)

        def init_state(self, n_neurons: int) -> Float[Array, "neurons state_plus1"]:
            inner = self.neuron.init_state(n_neurons)  # (N, S)
            r = jnp.zeros((n_neurons, 1), dtype=self.dtype)
            return jnp.concatenate([inner, r], axis=1)

        def _split_state(
            self, y: Float[Array, "neurons state_plus1"]
        ) -> tuple[Float[Array, "neurons inner_state"], Float[Array, "neurons"]]:
            inner = y[:, :-1]
            r = y[:, -1]
            return inner, r

        def _merge_state(
            self,
            inner: Float[Array, "neurons inner_state"],
            r: Float[Array, "neurons"],
        ) -> Float[Array, "neurons state_plus1"]:
            return jnp.concatenate([inner, r[:, None]], axis=1)

        def dynamics(
            self,
            t: float,
            y: Float[Array, "neurons state_plus1"],
            args: Dict[str, Any],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, r = self._split_state(y)
            d_inner = self.neuron.dynamics(t, inner, args)
            dr_dt = jnp.where(r > 0.0, -1.0, 0.0).astype(self.dtype)
            refrac = (r > 0.0).astype(self.dtype)[:, None]
            if self.refrac_mask is None:
                gate = 1.0 - refrac
            else:
                mask = self.refrac_mask.value
                if mask.ndim == 0:
                    mask = jnp.full((inner.shape[1],), mask, dtype=self.dtype)
                else:
                    if mask.shape[0] != inner.shape[1]:
                        raise ValueError(
                            f"`refrac_mask` length {mask.shape[0]} "
                            f"does not match state size {inner.shape[1]}"
                        )
                mask = mask[None, :].astype(self.dtype)
                gate = 1.0 - refrac * mask

            d_inner = d_inner * gate
            return self._merge_state(d_inner, dr_dt)

        def spike_condition(
            self,
            t: float,
            y: Float[Array, "neurons state_plus1"],
            **kwargs: Dict[str, Any],
        ) -> Float[Array, "neurons"]:
            inner, r = self._split_state(y)
            base_cond = self.neuron.spike_condition(t, inner, **kwargs)
            refractory = r > 0.0
            blocked_cond = -jnp.ones_like(base_cond)
            return jnp.where(refractory, blocked_cond, base_cond)

        def reset_spiked(
            self,
            y: Float[Array, "neurons state_plus1"],
            spike_mask: Bool[Array, "neurons"],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, r = self._split_state(y)
            inner_new = self.neuron.reset_spiked(inner, spike_mask)
            r_new = jnp.where(spike_mask, self.t_refrac.value, r)
            return self._merge_state(inner_new, r_new)

        def input_spike(
            self,
            y: Float[Array, "neurons state_plus1"],
            from_idx: Int[Array, ""],
            to_idx: Int[Array, "targets"],
            valid_mask: Bool[Array, "targets"],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, r = self._split_state(y)
            if self.block_input_during_refrac:
                post_active = (r[to_idx] <= 0.0)
                effective_mask = valid_mask & post_active
            else:
                effective_mask = valid_mask

            inner_new = self.neuron.input_spike(inner, from_idx, to_idx, effective_mask)
            return self._merge_state(inner_new, r)

    RefractoryNeuron.__name__ = f"Refractory_{NeuronCls.__name__}"
    return RefractoryNeuron
