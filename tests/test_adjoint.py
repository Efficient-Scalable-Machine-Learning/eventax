import jax
import jax.numpy as jnp
import optimistix as optx
import diffrax as dfx
import equinox as eqx

from eventax.evnn import EvNN
from eventax.neuron_models import LIF
from eventax.adjoint import EventPropAdjoint

_grad_ttfs = eqx.filter_jit(
    lambda snn: eqx.filter_grad(
        lambda m: m.ttfs(in_spike_times=jnp.array([[0.0]]))[0]
    )(snn)
)


def test_gradients():
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
            dtype=jnp.float32,
            adjoint=dfx.RecursiveCheckpointAdjoint(None)
        )

        grad_bptt = _grad_ttfs(snn)

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
            dtype=jnp.float32,
            adjoint=EventPropAdjoint()
        )

        grad_adjoint = _grad_ttfs(snn)

        filtered = eqx.filter(grad_bptt, eqx.is_array)
        arrays_bptt = jax.tree_util.tree_leaves(filtered)

        filtered = eqx.filter(grad_adjoint, eqx.is_array)
        arrays_adjoint = jax.tree_util.tree_leaves(filtered)

        comparisons = jax.tree_util.tree_map(
            lambda a, b: jnp.allclose(a, b, rtol=1e-3),
            arrays_bptt, arrays_adjoint
        )

        assert all(jax.tree_util.tree_leaves(comparisons)), (
            f"Grad mismatch for tmem={tmem:.3f}, ic={ic:.3f}, thresh={thresh:.3f}"
        )
