from __future__ import annotations

from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

ScalarOrArray = Union[int, float, jnp.ndarray]


def _fan_in_scale(
    wmask: Float[Array, "pre post"],
    mode: Optional[str],
    eps: float = 1e-8,
) -> Float[Array, "post"]:

    if mode is None:
        return jnp.ones(wmask.shape[1], dtype=wmask.dtype)

    fan_in = jnp.sum(wmask, axis=0)  # shape (post,)
    fan_in = jnp.maximum(fan_in, 1.0)  # avoid division by 0

    if mode == "sqrt":
        scale = 1.0 / jnp.sqrt(fan_in + eps)
    elif mode == "linear":
        scale = 1.0 / (fan_in + eps)
    else:
        raise ValueError(
            f"Unknown fan_in_mode '{mode}'. Use None, 'sqrt', or 'linear'."
        )

    return scale


def _init_weights(
    key: Optional[PRNGKeyArray],
    shape: Tuple[int, int],
    *,
    wlim: Optional[float],
    init_weights: Optional[ScalarOrArray],
    dtype=jnp.float32,
    wmask: Optional[Bool[Array, "pre post"]] = None,
    fan_in_mode: Optional[str] = None,
    wmean: ScalarOrArray = 0.0,
) -> Float[Array, "pre post"]:

    if init_weights is not None:
        if isinstance(init_weights, (int, float)):
            w = jnp.full(shape, init_weights, dtype=dtype)
        else:
            w = jnp.asarray(init_weights, dtype=dtype)

        if wmask is not None:
            w = w * wmask.astype(dtype)
        return w

    if key is None:
        raise ValueError("Must provide `key` when `init_weights` is None.")
    if wlim is None:
        raise ValueError("If `init_weights` is None, `wlim` must be set.")

    wmean_arr = jnp.asarray(wmean, dtype=dtype)
    half = jnp.asarray(wlim / 2.0, dtype=dtype)

    minval = wmean_arr - half
    maxval = wmean_arr + half

    base = jax.random.uniform(
        key, shape, dtype=dtype, minval=minval, maxval=maxval
    )

    if wmask is not None:
        wmask_f = wmask.astype(dtype)

        scale = _fan_in_scale(wmask_f, fan_in_mode)
        base = base * scale[None, :]

        base = base * wmask_f

    return base


def _init_bias(
    key: Optional[PRNGKeyArray],
    shape: Tuple[int, ...],
    *,
    blim: Optional[float],
    init_bias: Optional[ScalarOrArray],
    dtype=jnp.float32,
    bmean: ScalarOrArray = 0.0,
) -> Float[Array, "..."]:

    if init_bias is not None:
        if isinstance(init_bias, (int, float)):
            return jnp.full(shape, init_bias, dtype=dtype)
        else:
            return jnp.asarray(init_bias, dtype=dtype)

    if key is None:
        raise ValueError("Must provide `key` when `init_bias` is None.")
    if blim is None:
        raise ValueError("If `init_bias` is None, `blim` must be set.")

    bmean_arr = jnp.asarray(bmean, dtype=dtype)
    half = jnp.asarray(blim / 2.0, dtype=dtype)

    minval = bmean_arr - half
    maxval = bmean_arr + half

    return jax.random.uniform(
        key, shape, dtype=dtype, minval=minval, maxval=maxval
    )


def init_weights_and_bias(
    key: Optional[PRNGKeyArray],
    *,
    n_neurons: int,
    in_size: int,
    wlim: Optional[float] = None,
    init_weights: Optional[ScalarOrArray] = None,
    blim: Optional[float] = None,
    init_bias: Optional[ScalarOrArray] = None,
    dtype=jnp.float32,
    wmask: Optional[Bool[Array, "in_plus_neurons neurons"]] = None,
    fan_in_mode: Optional[str] = None,
    wmean: ScalarOrArray = 0.0,
    bmean: ScalarOrArray = 0.0,
) -> Tuple[
    Float[Array, "in_plus_neurons neurons"],
    Float[Array, "neurons"],
]:

    weights_shape = (n_neurons + in_size, n_neurons)
    bias_shape = (n_neurons,)

    need_w = init_weights is None
    need_b = init_bias is None

    if (need_w or need_b) and key is None:
        raise ValueError(
            "Must provide `key` when either weights or bias need random initialization."
        )

    if need_w and need_b:
        wkey, bkey = jax.random.split(key, 2)
    elif need_w:
        wkey, bkey = key, None
    elif need_b:
        wkey, bkey = None, key
    else:
        wkey = bkey = None

    weights = _init_weights(
        wkey,
        weights_shape,
        wlim=wlim,
        init_weights=init_weights,
        dtype=dtype,
        wmask=wmask,
        fan_in_mode=fan_in_mode,
        wmean=wmean,
    )

    bias = _init_bias(
        bkey,
        bias_shape,
        blim=blim,
        init_bias=init_bias,
        dtype=dtype,
        bmean=bmean,
    )

    return weights, bias
