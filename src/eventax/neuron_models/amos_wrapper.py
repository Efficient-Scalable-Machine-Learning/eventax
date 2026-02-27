import jax.numpy as jnp
from typing import Any, Dict, Type, TypeVar
from jaxtyping import Array, Float, Int, Bool
from .base_model import NeuronModel

TNeuron = TypeVar("TNeuron", bound=NeuronModel)


def AMOS(
    NeuronCls: Type[TNeuron],
    freeze_dynamics: bool = False,
    stop_incoming: bool = False,
) -> Type[NeuronModel]:
    class AMOSNeuron(NeuronModel):
        neuron: NeuronModel

        def __init__(self, *args, **kwargs):
            inner = NeuronCls(*args, **kwargs)
            super().__init__(dtype=inner.dtype)
            self.neuron = inner

        def init_state(self, n_neurons: int) -> Float[Array, "neurons state_plus1"]:
            inner = self.neuron.init_state(n_neurons)
            has_spiked = jnp.zeros((n_neurons, 1), dtype=self.dtype)
            return jnp.concatenate([inner, has_spiked], axis=1)

        def _split_state(
            self, y: Float[Array, "neurons state_plus1"]
        ) -> tuple[Float[Array, "neurons inner_state"], Float[Array, "neurons"]]:
            return y[:, :-1], y[:, -1]

        def _merge_state(
            self,
            inner: Float[Array, "neurons inner_state"],
            has_spiked: Float[Array, "neurons"],
        ) -> Float[Array, "neurons state_plus1"]:
            return jnp.concatenate([inner, has_spiked[:, None]], axis=1)

        def dynamics(
            self,
            t: float,
            y: Float[Array, "neurons state_plus1"],
            args: Dict[str, Any],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, has_spiked = self._split_state(y)
            d_inner = self.neuron.dynamics(t, inner, args)

            if freeze_dynamics:
                already_spiked = (has_spiked > 0.5)[:, None]
                d_inner = jnp.where(already_spiked, jnp.zeros_like(d_inner), d_inner)

            d_has_spiked = jnp.zeros_like(has_spiked)
            return self._merge_state(d_inner, d_has_spiked)

        def spike_condition(
            self,
            t: float,
            y: Float[Array, "neurons state_plus1"],
            **kwargs: Dict[str, Any],
        ) -> Float[Array, "neurons"]:
            inner, has_spiked = self._split_state(y)
            base_cond = self.neuron.spike_condition(t, inner, **kwargs)
            already_spiked = has_spiked > 0.5
            return jnp.where(already_spiked, -jnp.ones_like(base_cond), base_cond)

        def reset_spiked(
            self,
            y: Float[Array, "neurons state_plus1"],
            spiked_mask: Bool[Array, "neurons"],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, has_spiked = self._split_state(y)
            inner_new = self.neuron.reset_spiked(inner, spiked_mask)
            has_spiked_new = jnp.where(spiked_mask, 1.0, has_spiked)
            return self._merge_state(inner_new, has_spiked_new)

        def input_spike(
            self,
            y: Float[Array, "neurons state_plus1"],
            from_idx: Int[Array, ""],
            to_idx: Int[Array, "targets"],
            valid_mask: Bool[Array, "targets"],
        ) -> Float[Array, "neurons state_plus1"]:
            inner, has_spiked = self._split_state(y)

            if stop_incoming:
                post_mask = has_spiked[to_idx] < 0.5
                valid_mask = valid_mask & post_mask

            inner_new = self.neuron.input_spike(inner, from_idx, to_idx, valid_mask)
            return self._merge_state(inner_new, has_spiked)

    AMOSNeuron.__name__ = f"AMOS_{NeuronCls.__name__}"
    return AMOSNeuron
