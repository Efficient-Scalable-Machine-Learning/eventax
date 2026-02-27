from typing import Any, Dict, Tuple, Type, Optional

import json
import equinox as eqx
import jax
import jax.numpy as jnp

from eventax.evnn import EvNN, FFEvNN
from eventax.neuron_models import (
    NeuronModel,
    LIF,
    EIF,
    PLIF,
    QIF,
    PQIF,
    EGRU,
    Izhikevich,
    pIzhikevich,
    ALIF,
    MultiNeuronModel,
)

EVNN_REGISTRY: Dict[str, Type[EvNN]] = {
    "EvNN": EvNN,
    "FFEvNN": FFEvNN,
}

NEURON_MODEL_REGISTRY: Dict[str, Type[NeuronModel]] = {
    "NeuronModel": NeuronModel,
    "LIF": LIF,
    "EIF": EIF,
    "PLIF": PLIF,
    "QIF": QIF,
    "PQIF": PQIF,
    "EGRU": EGRU,
    "Izhikevich": Izhikevich,
    "pIzhikevich": pIzhikevich,
    "ALIF": ALIF,
    "MultiNeuronModel": MultiNeuronModel,
}

_EXCLUDE_KEYS = {"key", "neuron_model"}


def _dtype_to_string(dtype: Any) -> str:
    if dtype is jnp.float32:
        return "float32"
    if dtype is jnp.float64:
        return "float64"
    if dtype is jnp.float16:
        return "float16"
    return str(dtype)


def _dtype_from_string(s: str) -> jnp.dtype:
    if s == "float32":
        return jnp.float32
    if s == "float64":
        return jnp.float64
    if s == "float16":
        return jnp.float16
    return getattr(jnp, s)


def make_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Filter and adapt a config dict so it can be JSON-serialised."""
    out: Dict[str, Any] = {}

    for k, v in config.items():
        if k in _EXCLUDE_KEYS:
            continue

        if k == "dtype":
            out[k] = _dtype_to_string(v)
            continue

        if k == "neuron_models":
            try:
                names = []
                for cls in v:
                    if isinstance(cls, type) and issubclass(cls, NeuronModel):
                        names.append(cls.__name__)
                    else:
                        raise TypeError(
                            f"Expected NeuronModel subclasses in 'neuron_models', got {cls!r}"
                        )
                out["neuron_models"] = names
                continue
            except TypeError:
                pass

        if k == "neuron_type_ids":
            arr = jnp.asarray(v, dtype=jnp.int32)
            out["neuron_type_ids"] = arr.tolist()
            continue

        try:
            json.dumps(v)
        except TypeError:
            continue
        else:
            out[k] = v

    return out


def save_evnn(
    path: str,
    model: EvNN,
    config: Dict[str, Any],
    *,
    version: int = 1,
) -> None:
    evnn_cls_name = model.__class__.__name__
    neuron_model_cls_name = type(model.neuron_model).__name__

    serial_cfg = make_serializable_config(config)

    header = {
        "version": version,
        "evnn_cls": evnn_cls_name,
        "neuron_model_cls": neuron_model_cls_name,
        "config": serial_cfg,
    }

    with open(path, "wb") as f:
        f.write(json.dumps(header).encode("utf-8") + b"\n")
        eqx.tree_serialise_leaves(f, model)


def load_evnn(
    path: str,
    key: jax.Array,
    *,
    neuron_model: Optional[Type[NeuronModel]] = None,
    evnn_registry: Dict[str, Type[EvNN]] = EVNN_REGISTRY,
    neuron_model_registry: Dict[str, Type[NeuronModel]] = NEURON_MODEL_REGISTRY,
    **overrides,
) -> Tuple[EvNN, Dict[str, Any]]:
    """Load an EvNN/FFEvNN from disk.

    Any additional keyword arguments will override the saved config values.
    For example: load_evnn(path, key, max_event_steps=300)
    """
    with open(path, "rb") as f:
        header = json.loads(f.readline().decode("utf-8"))

        evnn_cls_name = header["evnn_cls"]
        header_neuron_model_cls_name = header["neuron_model_cls"]
        cfg = dict(header["config"])

        if "dtype" in cfg and isinstance(cfg["dtype"], str):
            cfg["dtype"] = _dtype_from_string(cfg["dtype"])

        if "neuron_type_ids" in cfg:
            cfg["neuron_type_ids"] = jnp.asarray(cfg["neuron_type_ids"], dtype=jnp.int32)

        if "neuron_models" in cfg:
            models = cfg["neuron_models"]
            if isinstance(models, (list, tuple)) and all(isinstance(m, str) for m in models):
                try:
                    cfg["neuron_models"] = [
                        neuron_model_registry[name] for name in models
                    ]
                except KeyError as e:
                    raise ValueError(
                        f"Unknown sub-neuron model '{e.args[0]}' in neuron_models."
                        "Add it to NEURON_MODEL_REGISTRY in io.py or adjust your config."
                    ) from e

        cfg.update(overrides)

        try:
            evnn_cls = evnn_registry[evnn_cls_name]
        except KeyError as e:
            raise ValueError(
                f"Unknown EvNN class '{evnn_cls_name}'. Add it to EVNN_REGISTRY in io.py."
            ) from e

        if neuron_model is not None:
            neuron_model_cls = neuron_model
        else:
            try:
                neuron_model_cls = neuron_model_registry[header_neuron_model_cls_name]
            except KeyError as e:
                raise ValueError(
                    f"Unknown NeuronModel class '{header_neuron_model_cls_name}' in file. "
                    "Pass neuron_model=YourNeuronModelClass to load_evnn to load custom models."
                ) from e

        template = evnn_cls(
            neuron_model=neuron_model_cls,
            key=key,
            **cfg,
        )

        model = eqx.tree_deserialise_leaves(f, template)

    return model, cfg
