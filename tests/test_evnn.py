import pytest
import jax
import jax.numpy as jnp
import optimistix as optx
import diffrax as dfx
import equinox as eqx

from eventax.evnn import EvNN, FFEvNN
from eventax.neuron_models import LIF

_eval_state = eqx.filter_jit(lambda snn, ct: snn.state_at_t(
    in_spike_times=jnp.array([[0.0]]), comp_times=jnp.array([ct])))

_eval_ttfs = eqx.filter_jit(lambda snn: snn.ttfs(
    in_spike_times=jnp.array([[0.0]])))

_grad_ttfs = eqx.filter_jit(
    lambda snn: eqx.filter_grad(
        lambda m: m.ttfs(in_spike_times=jnp.array([[0.0]]))[0]
    )(snn)
)


def test_dynamics_randomized():
    key = jax.random.PRNGKey(42)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    for _ in range(20):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        tsyn = float(jax.random.uniform(k1, (), minval=1.0, maxval=100.0))
        tmem = float(jax.random.uniform(k2, (), minval=1.0, maxval=100.0))
        if abs(tmem - tsyn) < 1e-3:
            tmem += 1.0

        comp_time = float(jax.random.uniform(k3, (), minval=1e-3, maxval=30.0))

        snn = EvNN(
            key=k4,
            neuron_model=LIF,
            n_neurons=1,
            max_solver_time=30,
            in_size=1,
            init_delays=0,
            solver_stepsize=0.1,
            root_finder=root_finder,
            solver=ode_solver,
            thresh=1.0,
            tsyn=tsyn,
            tmem=tmem,
            init_bias=0.0,
            init_weights=1.0,
        )

        state_over_times = _eval_state(snn, comp_time)

        # (time_idx, out_neuron, state)
        got_v = state_over_times[0, 0, 0]
        got_i = state_over_times[0, 0, 1]

        expected_i = jnp.exp(-comp_time / tsyn)
        expected_v = (
            jnp.exp(-comp_time / tmem) - jnp.exp(-comp_time / tsyn)
        ) / ((1 / tsyn - 1 / tmem) * tmem)

        assert jnp.isclose(got_i, expected_i, atol=1e-2), (
            f"I mismatch for tsyn={tsyn:.3f}, tmem={tmem:.3f}, t={comp_time:.3f}: "
            f"got {float(got_i):.6f}, expected {float(expected_i):.6f}"
        )
        assert jnp.isclose(got_v, expected_v, atol=1e-2), (
            f"V mismatch for tsyn={tsyn:.3f}, tmem={tmem:.3f}, t={comp_time:.3f}: "
            f"got {float(got_v):.6f}, expected {float(expected_v):.6f}"
        )


def test_spike_time_randomized():
    key = jax.random.PRNGKey(42)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    for _ in range(20):
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

        tsyn = float(jax.random.uniform(k1, (), minval=1.0, maxval=100.0))
        tmem = float(jax.random.uniform(k2, (), minval=1.0, maxval=100.0))
        ic = float(jax.random.uniform(k3, (), minval=0.2, maxval=4.0))
        t_max = 30.0

        thresh_max = float(ic * (1.0 - jnp.exp(-t_max / tmem)))
        thresh_max = max(thresh_max, 1e-6)
        thresh = float(jax.random.uniform(k4, (), minval=0.0, maxval=thresh_max))

        snn = EvNN(
            key=k5,
            neuron_model=LIF,
            n_neurons=1,
            max_solver_time=30,
            in_size=1,
            init_delays=0,
            solver_stepsize=0.1,
            root_finder=root_finder,
            solver=ode_solver,
            thresh=thresh,
            tsyn=tsyn,
            tmem=tmem,
            init_bias=ic,
            init_weights=0.0,
        )

        got_spike_time = _eval_ttfs(snn)[0]
        expected_spike_time = tmem * jnp.log(ic / (ic - thresh))

        assert jnp.isclose(got_spike_time, expected_spike_time, atol=1e-4), (
            f"Spike time mismatch for tsyn={tsyn:.3f}, tmem={tmem:.3f}, ic={ic:.3f}, "
            f"thresh={thresh:.3f}: got {float(got_spike_time):.6f}, "
            f"expected {float(expected_spike_time):.6f}"
        )


def test_gradient_ttfs():
    key = jax.random.PRNGKey(0)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    for _ in range(2):
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

        tsyn = float(jax.random.uniform(k1, (), minval=1.0, maxval=10.0))
        tmem = float(jax.random.uniform(k2, (), minval=1.0, maxval=10.0))
        ic = float(jax.random.uniform(k3, (), minval=0.2, maxval=4.0))
        thresh = float(jax.random.uniform(k4, (), minval=0.01, maxval=ic * 0.95))

        snn = EvNN(
            key=k5,
            neuron_model=LIF,
            n_neurons=1,
            max_solver_time=30,
            in_size=1,
            init_delays=0,
            solver_stepsize=0.1,
            root_finder=root_finder,
            solver=ode_solver,
            thresh=thresh,
            tsyn=tsyn,
            tmem=tmem,
            init_bias=ic,
            init_weights=0.0,
            dtype=jnp.float32
        )

        grads = _grad_ttfs(snn)
        # eqx.filter_grad returns pytree matching snn; bias is neuron_model.ic
        got_grad = grads.neuron_model.ic[0]
        expected_grad = tmem * (1 / ic - 1 / (ic - thresh))

        assert jnp.isclose(got_grad, expected_grad, rtol=1e-3), (
            f"Grad mismatch for tmem={tmem:.3f}, ic={ic:.3f}, thresh={thresh:.3f}: "
            f"got {float(got_grad):.6f}, expected {float(expected_grad):.6f}"
        )


def test_jit():
    key = jax.random.PRNGKey(123)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    snn = EvNN(
        key=key,
        neuron_model=LIF,
        n_neurons=1,
        max_solver_time=30,
        in_size=1,
        init_delays=0.1,
        solver_stepsize=0.1,
        root_finder=root_finder,
        solver=ode_solver,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.0,
        init_weights=1.0,
    )

    spike_times = jnp.array([[0.0, 0.5, 1.0]])
    out = snn.ttfs(spike_times)
    out_jit = jax.jit(lambda st: snn.ttfs(st))(spike_times)
    assert out.shape == out_jit.shape
    assert jnp.allclose(out, out_jit, atol=1e-6)

    comp_time = jnp.array([2.5])
    state = snn.state_at_t(in_spike_times=spike_times, comp_times=comp_time)
    state_jit = jax.jit(lambda st, t: snn.state_at_t(in_spike_times=st,
                        comp_times=jnp.array([t])[0]))(spike_times, comp_time)
    flat = jnp.concatenate([jnp.ravel(x) for x in state])
    flat_jit = jnp.concatenate([jnp.ravel(x) for x in state_jit])
    assert state.shape == state_jit.shape
    assert jnp.allclose(flat, flat_jit, atol=1e-6)


def test_batching():
    key = jax.random.PRNGKey(1234)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    snn = EvNN(
        key=key,
        neuron_model=LIF,
        n_neurons=1,
        max_solver_time=30,
        in_size=1,
        init_delays=0.1,
        solver_stepsize=0.1,
        root_finder=root_finder,
        solver=ode_solver,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=2.0,
        dtype=jnp.float32
    )

    batch_size = 5
    spike_times_batch = jnp.stack([
        jnp.array([[0.0, 0.1 * i, 0.2 * i, 0.3 * i]]) for i in range(1, batch_size + 1)
    ])
    comp_times = jnp.linspace(0.5, 3.5, batch_size)

    batched_ttfs = jax.vmap(lambda st: snn.ttfs(st))(spike_times_batch)
    expected_ttfs = jnp.stack([snn.ttfs(st) for st in spike_times_batch], axis=0)
    assert batched_ttfs.shape == expected_ttfs.shape
    assert jnp.allclose(batched_ttfs, expected_ttfs, atol=1e-6)

    batched_state = jax.vmap(
        lambda st, t: snn.state_at_t(in_spike_times=st, comp_times=jnp.array([t])),
        in_axes=(0, 0)
    )(spike_times_batch, comp_times)
    expected_state = jnp.stack([
        snn.state_at_t(in_spike_times=st, comp_times=jnp.array([t]))
        for st, t in zip(spike_times_batch, comp_times)
    ], axis=0)
    assert batched_state.shape == expected_state.shape
    assert jnp.allclose(batched_state, expected_state, atol=1e-6)


def test_max_event_step_error():

    key = jax.random.PRNGKey(1234)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    snn = EvNN(
        key=key,
        neuron_model=LIF,
        n_neurons=1,
        max_solver_time=30,
        in_size=1,
        init_delays=0.1,
        solver_stepsize=0.1,
        root_finder=root_finder,
        solver=ode_solver,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=2.0,
        dtype=jnp.float32,
        max_event_steps=3
    )
    times = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    with pytest.raises(Exception):
        snn.ttfs(times)
    with pytest.raises(Exception):
        snn.state_at_t(times, 3)


def test_no_delays():

    key = jax.random.PRNGKey(1234)

    weights = jax.random.uniform(key, shape=(27, 26), minval=0.0, maxval=5.0)

    snn1 = FFEvNN(
        key=key,
        neuron_model=LIF,
        layers=[10, 5, 8, 3],
        max_solver_time=30,
        in_size=1,
        solver_stepsize=0.1,
        max_event_steps=20000,
        buffer_capacity=1000,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=weights,
        init_delays=0.0,
        dtype=jnp.float32,
    )

    snn2 = FFEvNN(
        key=key,
        neuron_model=LIF,
        layers=[10, 5, 8, 3],
        max_solver_time=30,
        in_size=1,
        solver_stepsize=0.1,
        max_event_steps=20000,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=weights,
        dtype=jnp.float32,
    )

    assert snn1.delays is not None and snn1.use_delays is True
    assert snn2.delays is None and snn2.use_delays is False

    spike_times = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    assert jnp.allclose(snn1.ttfs(spike_times), snn2.ttfs(spike_times), atol=1e-9)
    check_times = jnp.linspace(0, 10, 500)
    assert jnp.allclose(snn1.state_at_t(spike_times, check_times), snn2.state_at_t(spike_times, check_times), atol=1e-9)


def test_axonal_delays():
    key = jax.random.PRNGKey(42)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    ode_solver = dfx.Tsit5()

    weights = jnp.ones((11, 10)) * 2.0
    axonal_delays = jax.random.uniform(key, shape=(11,), minval=0.1, maxval=1.0)

    synaptic_delays = jnp.broadcast_to(axonal_delays[:, None], (11, 10))

    snn_axonal = FFEvNN(
        key=key,
        neuron_model=LIF,
        layers=[5, 5],
        max_solver_time=50,
        in_size=1,
        solver_stepsize=0.1,
        max_event_steps=5000,
        buffer_capacity=1000,
        root_finder=root_finder,
        solver=ode_solver,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=weights,
        init_axonal_delays=axonal_delays,
        dtype=jnp.float32,
    )

    snn_synaptic = FFEvNN(
        key=key,
        neuron_model=LIF,
        layers=[5, 5],
        max_solver_time=50,
        in_size=1,
        solver_stepsize=0.1,
        max_event_steps=5000,
        buffer_capacity=1000,
        root_finder=root_finder,
        solver=ode_solver,
        thresh=1.0,
        tsyn=5.0,
        tmem=10.0,
        init_bias=0.5,
        init_weights=weights,
        init_delays=synaptic_delays,
        dtype=jnp.float32,
    )

    spike_times = jnp.array([[0.0, 0.5]])

    assert jnp.allclose(snn_axonal.ttfs(spike_times), snn_synaptic.ttfs(spike_times), atol=1e-6)

    comp_times = jnp.linspace(0, 10, 100)
    assert jnp.allclose(
        snn_axonal.state_at_t(spike_times, comp_times),
        snn_synaptic.state_at_t(spike_times, comp_times),
        atol=1e-6,
    )
