import jax
import jax.numpy as jnp
from eventax.neuron_models import LIF, QIF


def test_lif_initialization():
    key = jax.random.PRNGKey(0)
    model = LIF(key, n_neurons=5, in_size=3, wmask=jnp.ones((8, 5)),
                thresh=1.0, tmem=20.0, tsyn=5.0, blim=0.5, wlim=0.5)
    assert model.weights.shape == (8, 5)
    assert model.ic.shape == (5,)


def test_lif_init_state():
    model = LIF(None, 5, 3, jnp.ones((8, 5)), 1.0, 5.0, 20.0, init_weights=0.1, init_bias=0.1)
    state = model.init_state(5)
    assert state.shape == (5, 2)
    assert jnp.allclose(state, 0.0)

# TODO extend this to actually check dynamics not only shapes and non-finiteness


def test_lif_dynamics():
    model = LIF(None, 5, 3, jnp.ones((8, 5)), 1.0, 5.0, 20.0, init_weights=0.1, init_bias=0.1)
    state = jnp.zeros((5, 2))
    dy = model.dynamics(0.0, state, args={})
    assert dy.shape == (5, 2)
    assert jnp.all(jnp.isfinite(dy))


def test_lif_spike_condition():
    model = LIF(None, 3, 2, jnp.ones((5, 3)), thresh=1.0, tsyn=5.0, tmem=20.0, init_weights=0.1, init_bias=0.1)
    state = jnp.array([[0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])
    cond = model.spike_condition(0.0, state)
    assert cond.shape == (3,)
    assert jnp.allclose(cond[0], -0.1)
    assert jnp.all(cond[2] > 0.0)


def test_lif_input_spike():
    model = LIF(None, 2, 1, jnp.ones((3, 2)), 1.0, 5.0, 20.0, init_weights=1.0, init_bias=0.1)
    state = jnp.zeros((2, 2))
    updated = model.input_spike(state, from_idx=0, to_idx=1, valid_mask=jnp.asarray(True))
    assert jnp.all(updated == jnp.array([[0.0, 0.0], [0.0, 1.0]]))


def test_lif_reset_spiked():
    model = LIF(None, 3, 1, jnp.ones((4, 3)), 1.0, 5.0, 20.0,
                init_weights=0.1, init_bias=0.1, reset_grad_preserve=False)
    state = jnp.array([[1.2, 0.5], [0.9, 0.5], [1.1, 0.5]])
    mask = jnp.array([True, False, True])
    reset = model.reset_spiked(state, mask)
    assert jnp.all(reset == jnp.array([[model.vreset.value, 0.5], [0.9, 0.5], [model.vreset.value, 0.5]]))


def test_qif_initialization():
    key = jax.random.PRNGKey(0)
    model = QIF(key, n_neurons=5, in_size=3, wmask=jnp.ones((8, 5)), tmem=20.0, tsyn=5.0, blim=0.5, wlim=0.5)
    assert model.weights.shape == (8, 5)
    assert model.ic.shape == (5,)


def test_qif_init_state():
    model = QIF(None, 5, 3, jnp.ones((8, 5)), 5.0, 20.0, init_weights=0.1, init_bias=0.1)
    state = model.init_state(5)
    assert state.shape == (5, 2)
    assert jnp.allclose(state[:, 0], 0.5)
    assert jnp.allclose(state[:, 1], 0.0)

# TODO extend this to actually check dynamics not only shapes and non-finiteness


def test_qif_dynamics():
    model = QIF(None, 5, 3, jnp.ones((8, 5)), 5.0, 20.0, init_weights=0.1, init_bias=0.1)
    state = jnp.zeros((5, 2))
    dy = model.dynamics(0.0, state, args={})
    assert dy.shape == (5, 2)
    assert jnp.all(jnp.isfinite(dy))


def test_qif_spike_condition():
    model = QIF(None, 3, 2, jnp.ones((5, 3)), tsyn=5.0, tmem=20.0, init_weights=0.1, init_bias=0.1)
    state = jnp.array([[0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])
    cond = model.spike_condition(0.0, state)
    assert cond.shape == (3,)
    assert jnp.allclose(cond[0], -0.1)
    assert jnp.all(cond[2] > 0.0)


def test_qif_input_spike():
    model = QIF(None, 2, 1, jnp.ones((3, 2)), 5.0, 20.0, init_weights=1.0, init_bias=0.1)
    state = jnp.zeros((2, 2))
    updated = model.input_spike(state, from_idx=0, to_idx=1, valid_mask=jnp.asarray(True))
    assert jnp.all(updated == jnp.array([[0.0, 0.0], [0.0, 1.0]]))


def test_qif_reset_spiked():
    model = QIF(None, 3, 1, jnp.ones((4, 3)), 5.0, 20.0, init_weights=0.1, init_bias=0.1)
    state = jnp.array([[1.2, 0.5], [0.9, 0.5], [1.1, 0.5]])
    mask = jnp.array([True, False, True])
    reset = model.reset_spiked(state, mask)
    assert jnp.all(reset == jnp.array([[model.vreset.value, 0.5], [0.9, 0.5], [model.vreset.value, 0.5]]))
