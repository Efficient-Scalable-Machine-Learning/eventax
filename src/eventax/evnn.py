import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import optimistix as optx
import inspect
from diffrax import AbstractStepSizeController, AbstractAdjoint
from .buffer import SpikeBuffer
from .neuron_models import NeuronModel, MultiNeuronModel
from math import ceil
from typing import Optional, Tuple, Any, Type
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from jax import tree_util
import warnings


class EvNN(eqx.Module):
    syn_conn: Int[Array, "in_plus_neurons max_syn"]
    max_syn_conn: int = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    delays: Optional[Float[Array, "in_plus_neurons neurons"]]
    axonal_delays: Optional[Float[Array, "in_plus_neurons"]]
    neuron_model: NeuronModel
    n_neurons: int = eqx.field(static=True)
    buffer_capacity: int = eqx.field(static=True)
    max_solver_steps: int = eqx.field(static=True)
    max_solver_time: float = eqx.field(static=True)
    solver_stepsize: float = eqx.field(static=True)
    max_event_steps: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    output_no_spike_value: float = eqx.field(static=True)
    solver: dfx.AbstractSolver = eqx.field(static=True)
    stepsize_controller: AbstractStepSizeController = eqx.field(static=True)
    output_indices: Int[Array, "n_out"]
    input_indices: Int[Array, "n_in"]
    use_delays: bool = eqx.field(static=True)
    use_axonal_delays: bool = eqx.field(static=True)
    spike_buffer: Type[SpikeBuffer] = eqx.field(static=True)
    adjoint: Any = eqx.field(static=True)
    root_finder: Any = eqx.field(static=True)
    adjoint: AbstractAdjoint = eqx.field(static=True)

    def __init__(
        self,
        neuron_model: NeuronModel,
        n_neurons: int,
        max_solver_time: float,
        in_size: int,
        key: PRNGKeyArray = None,
        t0: float = 0.0,
        wmask: Float[Array, "in_plus_neurons neurons"] = None,
        init_delays: Float[Array, "in_plus_neurons neurons"] = None,
        dlim: float = None,
        init_axonal_delays: Float[Array, "in_plus_neurons"] = None,
        axonal_dlim: float = None,
        output_neurons=None,
        input_neurons=None,
        buffer_capacity: int | None = None,
        max_event_steps: int = 1000,
        solver_stepsize: float = 0.001,
        output_no_spike_value: float = jnp.inf,
        root_finder=None,
        stepsize_controller=None,
        solver=None,
        adjoint=None,
        dtype=jnp.float32,
        **neuron_model_kwargs,
    ) -> None:

        self.use_delays = False

        if init_delays is None:
            if dlim is None:
                self.delays = None
            else:
                if key is None:
                    raise ValueError(
                        "Must set key to randomly initialize delays because init_delays is None and dlim is set"
                    )
                key, dkey = jax.random.split(key)
                self.delays = jax.random.uniform(
                    dkey,
                    (n_neurons + in_size, n_neurons),
                    minval=0,
                    maxval=dlim,
                    dtype=dtype,
                )
                self.use_delays = True

        elif isinstance(init_delays, (int, float)):
            self.delays = jnp.full(
                (n_neurons + in_size, n_neurons),
                init_delays,
                dtype=dtype,
            )
            self.use_delays = True
        else:
            self.delays = init_delays
            self.use_delays = True

        # Initialize axonal delays
        self.use_axonal_delays = False

        if init_axonal_delays is None:
            if axonal_dlim is None:
                self.axonal_delays = None
            else:
                if key is None:
                    raise ValueError(
                        "Must set key to randomly initialize axonal delays because"
                        "init_axonal_delays is None and axonal_dlim is set"
                    )
                key, akey = jax.random.split(key)
                self.axonal_delays = jax.random.uniform(
                    akey,
                    (n_neurons + in_size,),
                    minval=0,
                    maxval=axonal_dlim,
                    dtype=dtype,
                )
                self.use_axonal_delays = True

        elif isinstance(init_axonal_delays, (int, float)):
            self.axonal_delays = jnp.full(
                (n_neurons + in_size,),
                init_axonal_delays,
                dtype=dtype,
            )
            self.use_axonal_delays = True
        else:
            self.axonal_delays = init_axonal_delays
            self.use_axonal_delays = True

        # Determine if any delays are used (for buffer capacity logic)
        any_delays = self.use_delays or self.use_axonal_delays

        if not any_delays:

            if buffer_capacity is None:
                buffer_capacity = 1

            elif buffer_capacity != 1:
                warnings.warn(
                    "No synaptic or axonal delays are used, so buffer_capacity is forced to 1. "
                    "For simulations without delays, buffer capacity should be 1.",
                    stacklevel=2,
                )
                buffer_capacity = 1
        else:

            if buffer_capacity is None:
                buffer_capacity = 1000

        key, neuron_key = jax.random.split(key)

        self.spike_buffer = SpikeBuffer

        self.neuron_model = neuron_model(
            key=neuron_key,
            n_neurons=n_neurons,
            in_size=in_size,
            wmask=wmask,
            dtype=dtype,
            **neuron_model_kwargs,
        )

        ids_dtype, no_ids_value = self.spike_buffer.calc_dtype_and_non_spike_value(
            n_neurons + in_size
        )

        if output_neurons is None:
            output_neurons = jnp.ones((n_neurons,))
        self.output_indices = jnp.array(
            jnp.where(output_neurons)[0],
            dtype=ids_dtype,
        )

        if input_neurons is None:
            input_neurons = jnp.ones((n_neurons,))
        self.input_indices = jnp.array(
            jnp.where(input_neurons)[0],
            dtype=ids_dtype,
        )

        if wmask is None:
            wmask = jnp.ones((n_neurons + in_size, n_neurons))
            input_rows = jnp.arange(n_neurons, n_neurons + in_size)
            wmask = wmask.at[input_rows, :].set(0)
            wmask = wmask.at[input_rows, self.input_indices].set(1)

        expected_wmask_shape = (n_neurons + in_size, n_neurons)
        if wmask.shape != expected_wmask_shape:
            raise ValueError(
                f"wmask must have shape {expected_wmask_shape}, but got {wmask.shape}"
            )

        if input_neurons.shape != (n_neurons,):
            raise ValueError(
                f"input_neurons must have shape {(n_neurons,)}, but got {input_neurons.shape}"
            )

        if output_neurons.shape != (n_neurons,):
            raise ValueError(
                f"output_neurons must have shape {(n_neurons,)}, but got {output_neurons.shape}"
            )

        # Create syn_conn for ALL neurons (including input neurons)
        syn_conn_list = [jnp.where(wmask[i])[0] for i in range(n_neurons + in_size)]
        self.max_syn_conn = max(x.shape[0] for x in syn_conn_list) if syn_conn_list else 0

        def pad1d(arr):
            return jnp.pad(
                arr,
                (0, self.max_syn_conn - arr.shape[0]),
                constant_values=no_ids_value,
            )

        self.syn_conn = (
            jnp.stack([pad1d(ids) for ids in syn_conn_list], axis=0)
            .astype(ids_dtype)
        )

        if output_no_spike_value is None:
            self.output_no_spike_value = jnp.inf
        else:
            self.output_no_spike_value = output_no_spike_value

        self.n_neurons = n_neurons

        self.buffer_capacity = buffer_capacity
        self.solver_stepsize = solver_stepsize
        self.max_event_steps = max_event_steps
        self.max_solver_time = max_solver_time
        self.in_size = in_size
        self.t0 = t0
        self.dtype = dtype

        self.max_solver_steps = ceil(max_solver_time / solver_stepsize) + 1

        if root_finder is None:
            self.root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        else:
            self.root_finder = root_finder

        if stepsize_controller is None:
            self.stepsize_controller = dfx.ConstantStepSize()
        else:
            self.stepsize_controller = stepsize_controller

        if solver is None:
            self.solver = dfx.Euler()
        else:
            self.solver = solver

        if adjoint is None:
            self.adjoint = dfx.RecursiveCheckpointAdjoint()
        else:
            self.adjoint = adjoint

    def _get_axonal_delay(self, neuron_idx):
        """Get axonal delay for a neuron, returning 0 if axonal delays are not used."""
        if self.use_axonal_delays:
            return jnp.maximum(self.axonal_delays[neuron_idx], 0.0)
        else:
            return 0.0

    def init_state(self) -> Any:
        state = self.neuron_model.init_state(self.n_neurons)

        def cast_leaf(x):
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(self.dtype)
            return x

        return tree_util.tree_map(cast_leaf, state)

    def init_buffer(
        self,
        in_spike_times: Optional[Float[Array, "in_size K"]],
        comp_times: Optional[Float[Array, "n_times"]] = None,
    ):
        _, non_spike_idx = self.spike_buffer.calc_dtype_and_non_spike_value(
            self.n_neurons + self.in_size
        )

        if in_spike_times is None:
            times = jnp.array([self.t0], dtype=self.dtype)
            from_indices = jnp.array([non_spike_idx])
            to_indices = jnp.array([0])

            if comp_times is not None:
                comp_times = jnp.ravel(comp_times).astype(self.dtype)
                n_times = comp_times.shape[0]
                comp_from = jnp.full((n_times,), non_spike_idx, dtype=from_indices.dtype)
                comp_to = jnp.full((n_times,), non_spike_idx, dtype=to_indices.dtype)

                times = jnp.concatenate([times, comp_times], axis=0)
                from_indices = jnp.concatenate([from_indices, comp_from], axis=0)
                to_indices = jnp.concatenate([to_indices, comp_to], axis=0)

            return self.spike_buffer.init(
                self.buffer_capacity,
                self.n_neurons,
                times,
                from_indices,
                to_indices,
                time_dtype=self.dtype,
            )

        if in_spike_times.ndim != 2 or in_spike_times.shape[0] != self.in_size:
            raise ValueError(
                f"EvNN expects (input size, K spikes per input) but got {in_spike_times.shape}"
            )

        M = self.in_size    # number of input slots
        K = in_spike_times.shape[1]  # max spikes per slot
        N = self.input_indices.shape[0]  # first layer dimension
        base = self.n_neurons

        from_range = jnp.arange(base, base + M)  # Indices of the input neurons start after n_neurons
        to_range = self.input_indices

        if self.use_delays:
            from_indices = jnp.repeat(from_range, N * K)
            to_indices = jnp.tile(to_range, M * K)

            times = in_spike_times.ravel()
            times = jnp.repeat(times, N)

            # Add synaptic delays
            times = times + jnp.maximum(self.delays[from_indices, to_indices], 0.0)

            # Add axonal delays if enabled
            if self.use_axonal_delays:
                times = times + jnp.maximum(self.axonal_delays[from_indices], 0.0)

            inf_mask = jnp.isinf(times)
            to_indices = jnp.where(inf_mask, non_spike_idx, to_indices)
            from_indices = jnp.where(inf_mask, non_spike_idx, from_indices)

        else:
            # Non-delay case: create pseudospikes with from_indices only
            from_indices = jnp.repeat(from_range, K)
            times = in_spike_times.ravel()

            # Add axonal delays if enabled (even without synaptic delays)
            if self.use_axonal_delays:
                axonal_delay_per_spike = jnp.maximum(self.axonal_delays[from_indices], 0.0)
                times = times + axonal_delay_per_spike

            # Use non_spike_idx for to_indices to indicate these are non-delay pseudospikes
            to_indices = jnp.full_like(from_indices, non_spike_idx)

            # mask out inf spikes
            inf_mask = jnp.isinf(times)
            from_indices = jnp.where(inf_mask, non_spike_idx, from_indices)

        # Add initial pseudospike for starting integration
        # This is distinct from non-delay pseudospikes (has from_idx=non_spike_idx, to_idx=0)
        times = jnp.concatenate((jnp.array([self.t0], dtype=self.dtype), times), axis=0)
        from_indices = jnp.concatenate((jnp.array([non_spike_idx]), from_indices), axis=0)
        to_indices = jnp.concatenate((jnp.array([0]), to_indices), axis=0)

        # Append comp_times as state_at_t pseudospikes if given
        if comp_times is not None:
            comp_times = jnp.ravel(comp_times).astype(self.dtype)
            n_times = comp_times.shape[0]
            comp_from = jnp.full((n_times,), non_spike_idx, dtype=from_indices.dtype)
            comp_to = jnp.full((n_times,), non_spike_idx, dtype=to_indices.dtype)

            times = jnp.concatenate([times, comp_times], axis=0)
            from_indices = jnp.concatenate([from_indices, comp_from], axis=0)
            to_indices = jnp.concatenate([to_indices, comp_to], axis=0)

        return self.spike_buffer.init(
            self.buffer_capacity,
            self.n_neurons + self.in_size,
            times,
            from_indices,
            to_indices,
            time_dtype=self.dtype,
        )

    def __call__(
        self,
        state: Any,  # PyTree
        buffer,
    ) -> Tuple[Float[Array, ""],
               Bool[Array, "neurons"],
               Any,
               eqx.Module]:
        """Integrate state between events (buffer spike or neuron spike)."""
        # peek at next event time
        t0, _, _ = self.spike_buffer.peek(buffer)

        def no_event(buf):
            # no spikes to integrate: return unchanged buffer
            return (
                jnp.minimum(t0, self.max_solver_time),
                jnp.zeros((self.n_neurons,), dtype=bool),
                state,
                buf,
            )

        def handle_event(buf):
            # pop one spike out, call it buf1
            t0, from_idx, to_idx, buf1 = self.spike_buffer.pop(buf)
            t1, _, _ = self.spike_buffer.peek(buf1)
            t1 = jnp.minimum(t1, self.max_solver_time)
            t0_clamped = jnp.minimum(t0, t1)

            # Handle different types of spikes/pseudospikes
            def handle_spike_input(args):
                s, f_idx, t_idx = args
                ns = buf.index_non_spike_value

                # Check if this is a non-delay spike (has from_idx, to_idx = non_spike)
                is_non_delay = (t_idx == ns) & (f_idx != ns)

                # Check if this is a state_at_t pseudospike (both indices are non_spike)
                is_state_pseudospike = (f_idx == ns) & (t_idx == ns)

                # Check if this is an init pseudospike (from_idx=non_spike, to_idx=0)
                is_init_pseudospike = (f_idx == ns) & (t_idx == 0)

                def process_non_delay():
                    # For non-delay: get all connections from the neuron and apply them
                    conn_to = self.syn_conn[f_idx]
                    valid = conn_to != ns
                    return self.neuron_model.input_spike(s, f_idx, conn_to, valid)

                def process_regular():
                    # Regular delayed spike
                    return self.neuron_model.input_spike(
                        s,
                        f_idx,
                        jnp.array([t_idx]),
                        jnp.array([True])
                    )

                def no_change():
                    return s

                # Chain of conditions to handle different spike types
                return jax.lax.cond(
                    is_state_pseudospike | is_init_pseudospike,
                    no_change,
                    lambda: jax.lax.cond(
                        is_non_delay,
                        process_non_delay,
                        process_regular,
                    ),
                )

            state1 = handle_spike_input((state, from_idx, to_idx))

            def integrate(buf_inner):

                def spike_cond(t, y, args, **kwargs):
                    return jnp.max(self.neuron_model.spike_condition(t, y)).astype(self.dtype)

                event = dfx.Event(spike_cond, self.root_finder, direction=True)

                # run ODE solve between t0 and t1 on state1
                sol = dfx.diffeqsolve(
                    dfx.ODETerm(self.neuron_model),
                    self.solver,
                    stepsize_controller=self.stepsize_controller,
                    t0=t0_clamped,
                    t1=t1,
                    dt0=self.solver_stepsize,
                    y0=state1,
                    event=event,
                    throw=True,
                    max_steps=self.max_solver_steps,
                    adjoint=self.adjoint,
                    saveat=dfx.SaveAt(t0=False, t1=True, steps=False, dense=False),
                )
                t_spike = sol.ts[-1]
                y_spike = tree_util.tree_map(lambda x: x[0], sol.ys) if sol.ts.shape[0] == 1 else sol.ys[-1]

                cond_now = self.neuron_model.spike_condition(t_spike, y_spike)
                # spike_mask = (jnp.max(cond_now) == cond_now) & sol.event_mask
                spiked = jnp.argmax(cond_now)
                spike_mask = jnp.zeros_like(cond_now, dtype=bool).at[spiked].set(sol.event_mask)
                state2 = self.neuron_model.reset_spiked(y_spike, spike_mask)

                # if any spikes, add them to buffer
                def add_spikes(op):
                    state_, b = op
                    spiked = jnp.argmax(spike_mask)

                    # NEW: get dtypes from internal_state
                    idx_dtype_from = b.internal_state.from_indices.dtype
                    idx_dtype_to = b.internal_state.to_indices.dtype

                    conn_to = self.syn_conn[spiked].astype(idx_dtype_to)
                    valid = conn_to != b.index_non_spike_value

                    ns = b.index_non_spike_value

                    # Get axonal delay for the spiking neuron (0 if not used)
                    axonal_delay = self._get_axonal_delay(spiked)

                    # Add pseudospike to continue integration at the right time
                    curr_time = jnp.array([t_spike])
                    curr_from = jnp.array([ns], dtype=idx_dtype_from)
                    curr_to = jnp.array([0], dtype=idx_dtype_to)

                    if self.use_delays:
                        # Synaptic delays + optional axonal delay
                        delayed_times = jnp.where(
                            valid,
                            t_spike + axonal_delay + jnp.maximum(self.delays[spiked, conn_to], 0.0),
                            jnp.inf,
                        )

                        delayed_from = jnp.where(valid, spiked, ns).astype(idx_dtype_from)
                        delayed_to = conn_to

                        # concat current time pseudospike and delayed times
                        all_times = jnp.concatenate((curr_time, delayed_times), axis=0)
                        all_from = jnp.concatenate((curr_from, delayed_from), axis=0)
                        all_to = jnp.concatenate((curr_to, delayed_to), axis=0)

                        # mask out spikes that exceed max_solver_time
                        time_mask = all_times < self.max_solver_time
                        all_times = jnp.where(time_mask, all_times, jnp.inf)
                        all_from = jnp.where(time_mask, all_from, ns)
                        all_to = jnp.where(time_mask, all_to, ns)

                        return state_, self.spike_buffer.add_multiple(b, all_times, all_from, all_to)

                    else:
                        # Non-delay case: add a pseudospike with the spiked neuron as from_idx
                        # Include axonal delay if enabled
                        spike_time = t_spike + axonal_delay
                        non_delay_from = jnp.array([spiked], dtype=idx_dtype_from)
                        non_delay_to = jnp.array([ns], dtype=idx_dtype_to)

                        # If axonal delay is non-zero, we need a continuation pseudospike
                        # at t_spike to keep integration going until the delayed spike arrives
                        if self.use_axonal_delays:
                            all_times = jnp.concatenate((curr_time, jnp.array([spike_time])), axis=0)
                            all_from = jnp.concatenate((curr_from, non_delay_from), axis=0)
                            all_to = jnp.concatenate((curr_to, non_delay_to), axis=0)
                            return state_, self.spike_buffer.add_multiple(b, all_times, all_from, all_to)
                        else:
                            return state_, self.spike_buffer.add(
                                b, spike_time, non_delay_from[0], non_delay_to[0]
                            )

                # check if any neuron spiked and if new spikes need to be generated
                any_spike = (jnp.sum(spike_mask.astype(jnp.int32)) > 0)
                state2, new_buf = jax.lax.cond(
                    any_spike,
                    add_spikes,
                    lambda op: op,
                    (state2, buf_inner),
                )
                return t_spike, spike_mask, state2, new_buf

            # choose between integrate or skip if t0 == t1
            return jax.lax.cond(
                t0_clamped < t1,
                integrate,
                lambda b: (t0_clamped, jnp.zeros((self.n_neurons,), bool), state1, b),
                buf1,
            )

        # if there is no spike in the buffer -> do no-op
        t_spike, spike_mask, state_final, buffer_final = jax.lax.cond(
            t0 != jnp.inf,
            handle_event,
            no_event,
            buffer,
        )

        return t_spike, spike_mask, state_final, buffer_final

    def ttfs(self, in_spike_times: Float[Array, "in_size K"]) -> Float[Array, "n_out"]:
        """For each output neuron returns the time it first fired for a given input."""

        in_spike_times = in_spike_times.astype(self.dtype)

        n_outputs = len(self.output_indices)

        def cond_fn(carry):
            t_curr, _, _, spike_buffer, first_spike_times_out = carry
            all_spiked = jnp.all(first_spike_times_out < self.output_no_spike_value)
            time_left = t_curr < self.max_solver_time
            return jnp.logical_and(jnp.logical_not(all_spiked), time_left)

        def body_fn(carry):
            t_spike, m_spike, state, spike_buffer, first_spike_times_out = carry

            t_spike_new, m_spike_new, state_new, spike_buffer_new = self(state, spike_buffer)

            first_spike_times_out_new = jnp.where(
                ((first_spike_times_out == self.output_no_spike_value) &
                 (m_spike_new[self.output_indices] > 0)),
                t_spike_new,
                first_spike_times_out,
            )

            return (t_spike_new, m_spike_new, state_new, spike_buffer_new, first_spike_times_out_new)

        init_carry = (
            0.0,
            jnp.zeros((self.n_neurons,), dtype=jnp.bool_),
            self.init_state(),
            self.init_buffer(in_spike_times),
            jnp.full((n_outputs,), self.output_no_spike_value, dtype=self.dtype),
        )

        out_carry = eqx.internal.while_loop(
            cond_fn, body_fn, init_carry, max_steps=self.max_event_steps, kind="bounded"
        )

        out_carry = eqx.error_if(
            out_carry, cond_fn(out_carry), "Reached max event steps. Try to increase event_steps."
        )

        _, _, _, _, first_spike_times = out_carry
        return first_spike_times

    def spikes_until_t(
        self,
        in_spike_times: Float[Array, "in_size K"],
        final_time: float,
        max_spikes: int = 100,
    ) -> Float[Array, "n_out max_spikes"]:
        n_outputs = len(self.output_indices)

        def cond_fn(carry):
            state, last_t, buffer, out_spikes, counter = carry
            max_spikes_reached = jnp.sum(counter) >= max_spikes
            empty_buffer = self.spike_buffer.is_empty(buffer)
            final_time_reached = last_t >= final_time
            return ~(max_spikes_reached | empty_buffer | final_time_reached)

        def body_fn(carry):
            state, _, buffer, out_spikes, counter = carry
            t_spike, m_spike, state_new, buffer_new = self(state, buffer)
            valid_time = t_spike <= final_time
            mask_out = m_spike[self.output_indices] & valid_time
            i = jnp.arange(n_outputs)
            slot = counter
            new_vals = jnp.where(mask_out, t_spike, out_spikes[i, slot])
            out_spikes = out_spikes.at[i, slot].set(new_vals)
            counter = counter + mask_out.astype(jnp.int32)
            return state_new, t_spike, buffer_new, out_spikes, counter

        init_state = self.init_state()
        init_buffer = self.init_buffer(in_spike_times)
        out_spikes = jnp.full((n_outputs, max_spikes), self.output_no_spike_value)
        init_counter = jnp.zeros((n_outputs,), dtype=jnp.int32)
        init_carry = (init_state, self.t0, init_buffer, out_spikes, init_counter)

        out_carry = eqx.internal.while_loop(
            cond_fn,
            body_fn,
            init_carry,
            max_steps=self.max_event_steps,
            kind="bounded",
        )
        out_carry = eqx.error_if(
            out_carry, cond_fn(out_carry), "Reached max event steps. Try to increase event_steps."
        )

        _, _, _, out_spikes, _ = out_carry

        return out_spikes

    def state_at_t(
        self,
        in_spike_times: Float[Array, "in_size K"],
        comp_times: Float[Array, "n_times"],
    ) -> Float[Array, "n_out n_times obs_channels"]:

        comp_times = jnp.ravel(comp_times)
        n_times = comp_times.shape[0]
        n_out = len(self.output_indices)

        init_state = self.init_state()
        sample_obs = self.neuron_model.observe(init_state)
        obs_dim = sample_obs.shape[-1]
        obs_dtype = sample_obs.dtype

        # init buffer with inputs + t0 pseudospike + comp_times pseudospikes
        buf = self.init_buffer(in_spike_times, comp_times)

        acc = jnp.full(
            (n_times, n_out, obs_dim),
            jnp.nan,
            dtype=obs_dtype,
        )

        def cond_fn(carry):
            _, buf, acc, _ = carry
            return jnp.any(jnp.isnan(acc)) & (~self.spike_buffer.is_empty(buf))

        def body_fn(carry):
            state, buf, acc, cnt = carry

            t, i1, i2 = self.spike_buffer.peek(buf)

            is_comp = jnp.logical_and(
                i1 == buf.index_non_spike_value,
                i2 == buf.index_non_spike_value,
            )

            def write_obs(a):
                obs = self.neuron_model.observe(state)
                return a.at[cnt].set(obs[self.output_indices])

            acc = jax.lax.cond(
                is_comp,
                write_obs,
                lambda a: a,
                acc,
            )

            _, _, state, buf = self(state, buf)

            cnt += is_comp

            return (state, buf, acc, cnt)

        init_carry = (init_state, buf, acc, 0)

        out_carry = eqx.internal.while_loop(
            cond_fn,
            body_fn,
            init_carry,
            max_steps=self.max_event_steps,
            kind="bounded",
        )

        out_carry = eqx.error_if(
            out_carry, cond_fn(out_carry), "Reached max event steps. Try to increase event_steps."
        )

        _, _, filled, _ = out_carry
        # transpose to (n_outputs, n_times, obs_dim)
        return filled.transpose((1, 0, 2))

    def record(self, in_spike_times: Float[Array, "in_size K"]):

        in_spike_times = in_spike_times.astype(self.dtype)

        init_state = self.init_state()
        buf0 = self.init_buffer(in_spike_times)
        idx_dtype = buf0.internal_state.from_indices.dtype
        no_id = buf0.index_non_spike_value

        def cond_fn(carry):
            t, m, state, buf, rec_t, rec_id, rec_buf, step = carry
            not_empty = jnp.logical_not(self.spike_buffer.is_empty(buf))
            steps_ok = step < self.max_event_steps
            return jnp.logical_and(not_empty, steps_ok)

        def body_fn(carry):
            t, m, state, buf, rec_t, rec_id, rec_buf, step = carry

            t_new, m_new, state_new, buf_new = self(state, buf)

            n_spikes_any = jnp.sum(m_new.astype(jnp.int32))
            did_spike = n_spikes_any == 1

            spike_id = jnp.argmax(m_new).astype(idx_dtype)

            rec_t = rec_t.at[step].set(jnp.where(did_spike, t_new, rec_t[step]))
            rec_id = rec_id.at[step].set(jnp.where(did_spike, spike_id, rec_id[step]))

            buf_size = jnp.asarray(self.spike_buffer.size(buf_new), dtype=jnp.int32)
            rec_buf = rec_buf.at[step].set(buf_size)

            return (t_new, m_new, state_new, buf_new, rec_t, rec_id, rec_buf, step + 1)

        init_carry = (
            jnp.array(0.0, dtype=self.dtype),
            jnp.zeros((self.n_neurons,), dtype=jnp.bool_),
            init_state,
            buf0,
            jnp.full((self.max_event_steps,), self.output_no_spike_value,
                     dtype=self.dtype),
            jnp.full((self.max_event_steps,), no_id, dtype=idx_dtype),
            jnp.full((self.max_event_steps,), -1, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )

        out = eqx.internal.while_loop(
            cond_fn, body_fn, init_carry, max_steps=self.max_event_steps, kind="bounded"
        )

        _, _, _, _, recorded_spike_times, recorded_spike_ids, recorded_buffer_sizes, _ = out
        return recorded_spike_times, recorded_spike_ids, recorded_buffer_sizes

    def get_wmask(self) -> Float[Array, "in_plus_neurons neurons"]:

        wmask = jnp.zeros((self.n_neurons + self.in_size, self.n_neurons), dtype=self.dtype)
        _, no_ids_value = self.spike_buffer.calc_dtype_and_non_spike_value(
            self.n_neurons + self.in_size
        )

        # Handle all neurons (including input neurons)
        valid = self.syn_conn != no_ids_value
        cols = jnp.where(valid, self.syn_conn, 0)
        rows = jnp.broadcast_to(
            jnp.arange(self.n_neurons + self.in_size)[:, None],
            self.syn_conn.shape,
        )

        rows_flat = rows[valid]
        cols_flat = cols[valid]
        wmask = wmask.at[rows_flat, cols_flat].set(1)

        return wmask


class FFEvNN(EvNN):
    """
    Feed-forward EvNN
    """

    def __init__(
        self,
        layers,
        in_size: int,
        neuron_model: NeuronModel,
        max_solver_time: float,
        key: PRNGKeyArray = None,
        init_delays: Float[Array, "in_plus_neurons neurons"] = None,
        dlim: float = None,
        init_axonal_delays: Float[Array, "in_plus_neurons"] = None,
        axonal_dlim: float = None,
        buffer_capacity: int | None = None,
        solver_stepsize: float = 0.01,
        adjoint=None,
        max_event_steps: int = 100,
        output_no_spike_value: float = None,
        root_finder=None,
        stepsize_controller=None,
        solver=None,
        dtype=jnp.float32,
        **neuron_model_kwargs,
    ):

        n_neurons = sum(layers)

        def create_feedforward_mask(layers, dtype):
            n_neurons_local = sum(layers)
            wmask_local = jnp.zeros((n_neurons_local, n_neurons_local), dtype=dtype)
            for layer in range(len(layers) - 1):
                s1 = sum(layers[:layer])
                s2 = s1 + layers[layer]
                s3 = s2 + layers[layer + 1]
                wmask_local = wmask_local.at[s1:s2, s2:s3].set(1)
            return wmask_local

        wmask = create_feedforward_mask(layers, dtype)

        output_neurons = jnp.zeros((n_neurons,), dtype=bool)
        output_neurons = output_neurons.at[-layers[-1]:].set(True)
        input_neurons = jnp.zeros((n_neurons,), dtype=bool)
        input_neurons = input_neurons.at[:layers[0]].set(True)

        wmask = jnp.concatenate(
            [wmask, jnp.zeros((in_size, n_neurons), dtype=wmask.dtype)],
            axis=0,
        )

        wmask = wmask.at[n_neurons:, input_neurons].set(1)

        # neuron model type mask building for multi-neuron model only
        is_multi_cls = inspect.isclass(neuron_model) and issubclass(neuron_model, MultiNeuronModel)
        is_multi_instance = isinstance(neuron_model, MultiNeuronModel)

        if (
            (is_multi_cls or is_multi_instance) and
            "neuron_type_ids" not in neuron_model_kwargs and
            "neuron_models" in neuron_model_kwargs and
            "neuron_model_kwargs" in neuron_model_kwargs
        ):
            submodels = neuron_model_kwargs["neuron_models"]
            submodel_kwargs_list = neuron_model_kwargs["neuron_model_kwargs"]

            if len(submodels) == len(layers) and len(submodel_kwargs_list) == len(layers):
                ids_per_layer = [
                    jnp.full((layers[i],), i, dtype=jnp.int32) for i in range(len(layers))
                ]
                neuron_type_ids = jnp.concatenate(ids_per_layer, axis=0)

                if neuron_type_ids.shape[0] != n_neurons:
                    raise ValueError(
                        f"Auto-generated neuron_type_ids length {neuron_type_ids.shape[0]} "
                        f"does not match n_neurons={n_neurons}."
                    )

                neuron_model_kwargs["neuron_type_ids"] = neuron_type_ids

        super().__init__(
            neuron_model=neuron_model,
            n_neurons=n_neurons,
            max_solver_time=max_solver_time,
            key=key,
            init_delays=init_delays,
            in_size=in_size,
            wmask=wmask,
            dlim=dlim,
            init_axonal_delays=init_axonal_delays,
            axonal_dlim=axonal_dlim,
            output_neurons=output_neurons,
            input_neurons=input_neurons,
            buffer_capacity=buffer_capacity,
            solver_stepsize=solver_stepsize,
            adjoint=adjoint,
            max_event_steps=max_event_steps,
            output_no_spike_value=output_no_spike_value,
            root_finder=root_finder,
            stepsize_controller=stepsize_controller,
            solver=solver,
            dtype=dtype,
            **neuron_model_kwargs,
        )
