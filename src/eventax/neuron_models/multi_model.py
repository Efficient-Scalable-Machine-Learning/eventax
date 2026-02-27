from typing import Any, Dict, Sequence, Mapping, Type, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray

from .base_model import NeuronModel, StaticArray


class MultiNeuronModel(NeuronModel):
    """Combine multiple neuron models via per-neuron type assignment.

    - Internal state is a PyTree: tuple[state_sub_0, state_sub_1, ...]
    - Each submodel has its own state shape (e.g. QIF: (n_qif, 2), ReLIF: (n_refrac, 3)).
    - spike_condition / observe return dense global arrays of shape (n_neurons, ...).
    """

    n_neurons: int = eqx.field(static=True)
    n_submodels: int = eqx.field(static=True)

    masks: Tuple[StaticArray, ...] = eqx.field(static=True)
    global_indices: Tuple[StaticArray, ...] = eqx.field(static=True)
    local_index: Tuple[StaticArray, ...] = eqx.field(static=True)
    sub_n_neurons: Tuple[int, ...] = eqx.field(static=True)

    submodels: Tuple[NeuronModel, ...]

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: int,
        in_size: int,
        wmask: Float[Array, "in_plus_neurons neurons"],
        *,
        neuron_models: Sequence[Type[NeuronModel]],
        neuron_model_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
        neuron_type_ids: Array,
        dtype=jnp.float32,
        **_ignored,
    ):
        super().__init__(dtype=dtype)

        self.n_neurons = n_neurons
        self.n_submodels = len(neuron_models)

        if neuron_model_kwargs is None:
            neuron_model_kwargs = [{} for _ in neuron_models]
        if len(neuron_model_kwargs) != self.n_submodels:
            raise ValueError("len(neuron_model_kwargs) must match len(neuron_models).")

        if neuron_type_ids is None:
            raise ValueError("Must pass `neuron_type_ids` (masks are no longer supported).")

        type_ids = jnp.asarray(neuron_type_ids, dtype=jnp.int32)
        if type_ids.shape != (n_neurons,):
            raise ValueError(
                f"`neuron_type_ids` must have shape ({n_neurons},), got {type_ids.shape}."
            )

        in_lower = jnp.all(type_ids >= 0)
        in_upper = jnp.all(type_ids < self.n_submodels)
        if not bool(in_lower & in_upper):
            raise ValueError("All neuron_type_ids must be in [0, n_submodels-1].")

        # Build masks internally from IDs
        masks_list = [type_ids == i for i in range(self.n_submodels)]

        stacked_masks = jnp.stack(masks_list, axis=0)
        per_neuron_count = stacked_masks.astype(jnp.int32).sum(axis=0)
        if not bool(jnp.all(per_neuron_count == 1)):
            raise ValueError(
                "Neuron types must form a partition: each neuron must belong to exactly one type."
            )

        masks_static: list[StaticArray] = []
        global_indices_static: list[StaticArray] = []
        local_index_list: list[jnp.ndarray] = []
        sub_n_list: list[int] = []

        for mask in masks_list:
            mask = jnp.asarray(mask, dtype=bool)
            global_ids = jnp.where(mask)[0]
            n_sub = int(global_ids.shape[0])

            indices = jnp.zeros((n_neurons,), dtype=jnp.int32)
            indices = indices.at[global_ids].set(jnp.arange(n_sub, dtype=jnp.int32))

            masks_static.append(StaticArray(mask))
            global_indices_static.append(StaticArray(global_ids))
            local_index_list.append(indices)
            sub_n_list.append(n_sub)

        self.masks = tuple(masks_static)
        self.global_indices = tuple(global_indices_static)
        self.local_index = tuple(StaticArray(idx) for idx in local_index_list)
        self.sub_n_neurons = tuple(sub_n_list)

        in_plus_global = wmask.shape[0]
        keys = jax.random.split(key, self.n_submodels)
        submodels: list[NeuronModel] = []

        for i, (model_cls, sub_kwargs) in enumerate(zip(neuron_models, neuron_model_kwargs)):
            mask = masks_list[i]
            n_sub = self.sub_n_neurons[i]

            wmask_sub = wmask[:, mask]

            in_size_local = in_plus_global - n_sub

            submodel = model_cls(
                key=keys[i],
                n_neurons=n_sub,
                in_size=in_size_local,
                wmask=wmask_sub,
                dtype=dtype,
                **sub_kwargs,
            )
            submodels.append(submodel)

        self.submodels = tuple(submodels)

    def init_state(self, n_neurons: int) -> Tuple[Any, ...]:
        if n_neurons != self.n_neurons:
            raise ValueError(f"Expected n_neurons={self.n_neurons}, got {n_neurons}.")

        states = []
        for submodel, n_sub in zip(self.submodels, self.sub_n_neurons):
            sub_state = submodel.init_state(n_sub)
            states.append(sub_state)
        return tuple(states)

    def dynamics(self, t: float, y: Any, args: Dict[str, Any]) -> Any:
        dy_list = []
        for sub_state, submodel in zip(y, self.submodels):
            dy = submodel.dynamics(t, sub_state, args)
            dy_list.append(dy)
        return tuple(dy_list)

    def spike_condition(
        self,
        t: float,
        y: Any,
        **kwargs: Dict[str, Any],
    ) -> Float[Array, "neurons"]:
        cond = jnp.zeros((self.n_neurons,), dtype=self.dtype)

        for sub_state, submodel, gids in zip(y, self.submodels, self.global_indices):
            c_sub = submodel.spike_condition(t, sub_state, **kwargs)  # (n_sub,)
            cond = cond.at[gids.value].set(c_sub)

        return cond

    def input_spike(
        self,
        y: Tuple[Any, ...],
        from_idx: Int[Array, ""],
        to_idx: Int[Array, "targets"],
        valid_mask: Bool[Array, "targets"],
    ) -> Tuple[Any, ...]:

        new_states = []

        for sub_state, submodel, mask_static, local_idx_static in zip(
            y, self.submodels, self.masks, self.local_index
        ):
            mask = mask_static.value
            local_idx = local_idx_static.value

            is_type = mask[to_idx]
            local_valid = valid_mask & is_type
            to_idx_local = local_idx[to_idx]

            sub_state_new = submodel.input_spike(
                y=sub_state,
                from_idx=from_idx,
                to_idx=to_idx_local,
                valid_mask=local_valid,
            )
            new_states.append(sub_state_new)

        return tuple(new_states)

    def reset_spiked(
        self,
        y: Tuple[Any, ...],
        spiked_mask: Bool[Array, "neurons"],
    ) -> Tuple[Any, ...]:

        new_states = []
        for sub_state, submodel, mask_static in zip(y, self.submodels, self.masks):
            mask = mask_static.value
            local_spikes = spiked_mask[mask]
            sub_state_new = submodel.reset_spiked(sub_state, local_spikes)
            new_states.append(sub_state_new)
        return tuple(new_states)

    def observe(
        self,
        y: Tuple[Any, ...],
    ) -> Float[Array, "neurons obs"]:

        obs_sub_list = []
        for sub_state, submodel in zip(y, self.submodels):
            obs_sub = submodel.observe(sub_state)
            obs_sub_list.append(obs_sub)

        obs_dims = [obs.shape[1] for obs in obs_sub_list]
        obs_dim = int(max(obs_dims))
        dtype = obs_sub_list[0].dtype

        obs_full = jnp.full((self.n_neurons, obs_dim), jnp.nan, dtype=dtype)

        for obs_sub, gids in zip(obs_sub_list, self.global_indices):
            g = gids.value
            d = obs_sub.shape[1]
            if d < obs_dim:
                pad_width = ((0, 0), (0, obs_dim - d))
                obs_sub = jnp.pad(obs_sub, pad_width, constant_values=jnp.nan)
            obs_full = obs_full.at[g].set(obs_sub)

        return obs_full
