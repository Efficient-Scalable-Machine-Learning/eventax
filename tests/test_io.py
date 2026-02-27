from eventax.io import save_evnn, load_evnn
from typing import Type, Any, Dict, List

import pytest
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from eventax.evnn import EvNN, FFEvNN

from eventax.neuron_models import (
    NeuronModel,
    LIF,
    EIF,
    QIF,
    Izhikevich,
    MultiNeuronModel,
)


def trees_allclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    a_leaves, _ = jtu.tree_flatten(a)
    b_leaves, _ = jtu.tree_flatten(b)

    if len(a_leaves) != len(b_leaves):
        return False

    for x, y in zip(a_leaves, b_leaves):
        if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
            if not bool(jnp.allclose(x, y, rtol=rtol, atol=atol)):
                return False
        else:
            eq = (x == y)
            if isinstance(eq, jnp.ndarray):
                if not bool(jnp.all(eq)):
                    return False
            else:
                if not bool(eq):
                    return False

    return True


def check_ttfs_equal(
    model1: EvNN,
    model2: EvNN,
    *,
    in_size: int,
    max_solver_time: float,
    dtype,
) -> None:
    base_times = jnp.array([0.1, 0.2, 0.3], dtype=dtype)
    in_spike_times = jnp.tile(base_times[None, :], (in_size, 1))

    tt1 = model1.ttfs(in_spike_times)
    tt2 = model2.ttfs(in_spike_times)

    if tt1.shape != tt2.shape:
        raise AssertionError(f"ttfs shape mismatch: {tt1.shape} vs {tt2.shape}")

    finite1 = jnp.isfinite(tt1)
    finite2 = jnp.isfinite(tt2)

    if not bool(jnp.any(finite1)):
        raise AssertionError("Original model produced no finite ttfs on the test input.")
    if not bool(jnp.any(finite2)):
        raise AssertionError("Loaded model produced no finite ttfs on the test input.")

    if bool(jnp.any(jnp.isnan(tt1))):
        raise AssertionError("Original model produced NaN ttfs.")
    if bool(jnp.any(jnp.isnan(tt2))):
        raise AssertionError("Loaded model produced NaN ttfs.")

    if not bool(jnp.allclose(tt1, tt2)):
        raise AssertionError("Forward ttfs outputs differ between original and loaded models.")


SINGLE_NEURON_MODELS: List[Type[NeuronModel]] = [
    LIF,
    EIF,
    QIF,
    Izhikevich,
]

NEURON_MODEL_KWARGS: Dict[Type[NeuronModel], Dict[str, Any]] = {
    LIF: {
        "thresh": 1.0,
        "tsyn": 5.0,
        "tmem": 10.0,
        "vreset": 0.0,
        "blim": 0.5,
        "wlim": 5.0,
        "init_bias": 0.0,
        "bmean": 0.25,
        "wmean": 2.5,
    },
    QIF: {
        "tmem": 10.0,
        "tsyn": 5.0,
        "blim": 1.5,
        "wlim": 25.0,
        "bmean": 0.75,
        "wmean": 12.5,
    },
    EIF: {
        "tmem": 10.0,
        "tsyn": 5.0,
        "vT": -50.0,
        "deltaT": 2.0,
        "EL": -65.0,
        "vreset": -65.0,
        "blim": 1.5,
        "wlim": 45.0,
        "bmean": 1.75,
        "wmean": 30.5,
    },
    Izhikevich: {
        "wlim": 5.0,
        "blim": 0.5,
        "init_bias": 0.0,
        "bmean": 0.25,
        "wmean": 2.5,
    },
}

""" TODO: Find reasonable params for EGRU to test
EGRU: {
    "tsyn": 5.0,
    "tmem": 20.0,
    "thresh": 0.0,
    "bias_scale": 0.1,
    "bias_mean": -2.0,
    "weight_scale": 0.1,
    "weight_mean": 0.2,
},
"""


def make_ffevnn_config(neuron_model_cls: Type[NeuronModel]) -> dict:
    cfg = {
        "key": jax.random.PRNGKey(0),
        "neuron_model": neuron_model_cls,
        "layers": [4, 3],
        "max_solver_time": 30.0,
        "in_size": 2,
        "solver_stepsize": 0.01,
        "max_event_steps": 500,
        "dtype": jnp.float32,
    }

    cfg.update(NEURON_MODEL_KWARGS.get(neuron_model_cls, {}))
    return cfg


@pytest.mark.parametrize("neuron_model_cls", SINGLE_NEURON_MODELS)
def test_save_load_single_neuron_models(tmp_path, neuron_model_cls: Type[NeuronModel]):
    cfg = make_ffevnn_config(neuron_model_cls)
    model = FFEvNN(**cfg)

    path = tmp_path / f"model_{neuron_model_cls.__name__}.eqx"
    save_evnn(str(path), model, cfg)

    key2 = jax.random.PRNGKey(1)
    loaded, loaded_cfg = load_evnn(
        str(path),
        key=key2,
        neuron_model=neuron_model_cls,
    )

    assert trees_allclose(model, loaded), f"Pytree mismatch for {neuron_model_cls.__name__}"
    check_ttfs_equal(
        model,
        loaded,
        in_size=cfg["in_size"],
        max_solver_time=cfg["max_solver_time"],
        dtype=cfg["dtype"],
    )


def make_multineuron_config() -> dict:
    layers = [3, 3]

    neuron_type_ids = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

    submodels: List[Type[NeuronModel]] = [LIF, EIF]

    sub_kwargs = [NEURON_MODEL_KWARGS.get(cls, {}) for cls in submodels]

    return {
        "key": jax.random.PRNGKey(42),
        "neuron_model": MultiNeuronModel,
        "layers": layers,
        "max_solver_time": 30.0,
        "in_size": 2,
        "solver_stepsize": 0.01,
        "max_event_steps": 500,
        "dtype": jnp.float32,
        "neuron_models": submodels,
        "neuron_model_kwargs": sub_kwargs,
        "neuron_type_ids": neuron_type_ids,
    }


def test_save_load_multineuron(tmp_path):
    cfg = make_multineuron_config()
    model = FFEvNN(**cfg)

    path = tmp_path / "model_multineuron.eqx"
    save_evnn(str(path), model, cfg)

    key2 = jax.random.PRNGKey(7)
    loaded, loaded_cfg = load_evnn(
        str(path),
        key=key2,
        neuron_model=MultiNeuronModel,
    )

    assert trees_allclose(model, loaded), "Pytree mismatch for MultiNeuronModel"
    check_ttfs_equal(
        model,
        loaded,
        in_size=cfg["in_size"],
        max_solver_time=cfg["max_solver_time"],
        dtype=cfg["dtype"],
    )
