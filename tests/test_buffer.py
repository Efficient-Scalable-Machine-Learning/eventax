import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from eventax.buffer import SpikeBuffer as SB


def test_init_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    assert jnp.all(SB.get_times(buffer) == jnp.array([0.3, 0.5, 0.6]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([1, 3, 2]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([2, 4, 3]))

    buffer = SB.init(buffer_size=30, n_neurons=10)

    assert jnp.all(SB.get_times(buffer) == jnp.array([]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([]))


def test_pop_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    t, from_idx, to_idx, buffer = SB.pop(buffer)

    assert t == 0.3
    assert from_idx == 1
    assert to_idx == 2
    assert jnp.all(SB.get_times(buffer) == jnp.array([0.5, 0.6]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([3, 2]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([4, 3]))


def test_pop_jit_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    jit_pop = jax.jit(SB.pop)
    t, from_idx, to_idx, buffer = jit_pop(buffer)

    assert t == 0.3
    assert from_idx == 1
    assert to_idx == 2
    assert jnp.all(SB.get_times(buffer) == jnp.array([0.5, 0.6]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([3, 2]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([4, 3]))


def test_add_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    buffer = SB.add(buffer, 0.4, 4, 5)

    assert jnp.all(SB.get_times(buffer) == jnp.array([0.3, 0.4, 0.5, 0.6]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([1, 4, 3, 2]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([2, 5, 4, 3]))


def test_add_jit_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    jit_add = jax.jit(SB.add)
    buffer = jit_add(buffer, 0.4, 4, 5)

    assert jnp.all(SB.get_times(buffer) == jnp.array([0.3, 0.4, 0.5, 0.6]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([1, 4, 3, 2]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([2, 5, 4, 3]))


def test_add_multiple_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    times = jnp.array([0.4, 0.55, 0.32, 0.8])
    from_ids = jnp.array([4, 7, 2, 1])
    to_ids = jnp.array([5, 8, 0, 2])

    buffer = SB.add_multiple(buffer, times, from_ids, to_ids)

    assert jnp.all(SB.get_times(buffer) == jnp.array([0.3, 0.32, 0.4, 0.5, 0.55, 0.6, 0.8]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([1, 2, 4, 3, 7, 2, 1]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([2, 0, 5, 4, 8, 3, 2]))


def test_add_multiple_jit_spikebuffer():

    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    times = jnp.array([0.4, 0.55, 0.32, 0.8])
    from_ids = jnp.array([4, 7, 2, 1])
    to_ids = jnp.array([5, 8, 0, 2])

    jit_add_multiple = jax.jit(SB.add_multiple)
    buffer = jit_add_multiple(buffer, times, from_ids, to_ids)

    assert jnp.all(SB.get_times(buffer) == jnp.array([0.3, 0.32, 0.4, 0.5, 0.55, 0.6, 0.8]))
    assert jnp.all(SB.get_from_indices(buffer) == jnp.array([1, 2, 4, 3, 7, 2, 1]))
    assert jnp.all(SB.get_to_indices(buffer) == jnp.array([2, 0, 5, 4, 8, 3, 2]))


def test_fill_and_empty_buffer_spikebuffer():

    key = jax.random.PRNGKey(42)
    n_neurons = 10

    buffer = SB.init(
        buffer_size=5000,
        n_neurons=n_neurons,
        times=jnp.array([]),
        from_indices=jnp.array([]),
        to_indices=jnp.array([]),
    )

    def cond_fn(carry):
        buffer, _ = carry
        return jnp.logical_not(SB.is_full(buffer))

    def body_fn(carry):
        buffer, key = carry
        key, tikey, frkey, tokey = jax.random.split(key, 4)
        random_times = jnp.abs(jax.random.normal(tikey, (50,)))
        random_from = jax.random.randint(
            frkey,
            (50,),
            0,
            n_neurons - 1,
            dtype=buffer.internal_state.from_indices.dtype,
        )
        random_to = jax.random.randint(
            tokey,
            (50,),
            0,
            n_neurons - 1,
            dtype=buffer.internal_state.to_indices.dtype,
        )
        return (SB.add_multiple(buffer, random_times, random_from, random_to), key)

    buffer, _ = eqx.internal.while_loop(
        cond_fn, body_fn, (buffer, key), max_steps=150, kind="bounded"
    )

    internal_times = buffer.internal_state.times
    assert jnp.all(internal_times != jnp.inf)
    assert SB.is_full(buffer)

    def cond_fn2(carry):
        buffer = carry
        return jnp.logical_not(SB.is_empty(buffer))

    def body_fn2(carry):
        buffer = carry
        _, _, _, buffer = SB.pop(buffer)
        return buffer

    buffer = eqx.internal.while_loop(
        cond_fn2, body_fn2, buffer, max_steps=5500, kind="bounded"
    )

    assert SB.is_empty(buffer)
    assert SB.size(buffer) == 0


def test_grad_through_add_spikebuffer():
    init_times = jnp.array([0.3, 0.6, 0.5])
    init_from = jnp.array([1, 2, 3])
    init_to = jnp.array([2, 3, 4])
    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=init_times,
        from_indices=init_from,
        to_indices=init_to,
    )

    n0 = init_times.shape[0]

    def f(ts_new):
        buf2 = SB.add_multiple(
            buffer,
            ts_new,
            jnp.zeros_like(ts_new, dtype=buffer.internal_state.from_indices.dtype),
            jnp.ones_like(ts_new, dtype=buffer.internal_state.to_indices.dtype),
        )
        all_t = jnp.concatenate(
            [buf2.external_state.times, buf2.internal_state.times]
        )
        perm = jnp.argsort(all_t)
        sorted_t = all_t[perm]
        k = n0 + ts_new.shape[0]
        return jnp.sum(sorted_t[:k])

    grad_f = jax.grad(f)
    g = grad_f(jnp.array([0.42]))
    assert g[0] == pytest.approx(1.0)

    jit_grad_f = jax.jit(grad_f)
    g2 = jit_grad_f(jnp.array([0.42]))
    assert g2[0] == pytest.approx(1.0)


def test_grad_through_init_and_pop_spikebuffer():
    init_times = jnp.array([0.6, 0.3, 0.5])
    init_from = jnp.array([1, 2, 3])
    init_to = jnp.array([2, 3, 4])

    def f(times):
        buf = SB.init(
            buffer_size=30,
            n_neurons=10,
            times=times,
            from_indices=init_from,
            to_indices=init_to,
        )
        t, _, _, _ = SB.pop(buf)
        return t

    grad_f = jax.grad(f)
    g = grad_f(init_times)
    assert jnp.all(g == jnp.array([0.0, 1.0, 0.0]))

    jit_grad_f = jax.jit(grad_f)
    g2 = jit_grad_f(init_times)
    assert jnp.all(g2 == jnp.array([0.0, 1.0, 0.0]))


def test_grad_through_add_multiple_spikebuffer():
    init_times = jnp.array([0.3, 0.6, 0.5])
    init_from = jnp.array([1, 2, 3])
    init_to = jnp.array([2, 3, 4])
    buffer = SB.init(
        buffer_size=30,
        n_neurons=10,
        times=init_times,
        from_indices=init_from,
        to_indices=init_to,
    )

    n0 = init_times.shape[0]
    new_times = jnp.array([0.42, 0.15, 0.78])

    def f(ts_new):
        buf2 = SB.add_multiple(
            buffer,
            ts_new,
            jnp.zeros_like(ts_new, dtype=buffer.internal_state.from_indices.dtype),
            jnp.ones_like(ts_new, dtype=buffer.internal_state.to_indices.dtype),
        )
        all_t = jnp.concatenate(
            [buf2.external_state.times, buf2.internal_state.times]
        )
        perm = jnp.argsort(all_t)
        sorted_t = all_t[perm]
        k = n0 + ts_new.shape[0]
        return jnp.sum(sorted_t[:k])

    grad_f = jax.grad(f)
    g = grad_f(new_times)
    assert jnp.all(g == jnp.ones_like(new_times))

    jit_grad_f = jax.jit(grad_f)
    g2 = jit_grad_f(new_times)
    assert jnp.all(g2 == jnp.ones_like(new_times))


def test_overflow_spikebuffer():

    buffer = SB.init(
        buffer_size=3,
        n_neurons=20,
        times=jnp.array([0.3, 0.6, 0.5]),
        from_indices=jnp.array([1, 2, 3]),
        to_indices=jnp.array([2, 3, 4]),
    )

    times = jnp.array([0.4, 0.55, 0.32, 0.8])
    from_ids = jnp.array([4, 7, 2, 1])
    to_ids = jnp.array([5, 8, 0, 2])

    with pytest.raises(Exception):
        _ = SB.add_multiple(buffer, times, from_ids, to_ids)
