import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Float, Int, Bool

TimeVec = Float[Array, "buffer"]
IdxVec = Int[Array, "buffer"]
TimeSc = Float[Array, ""]
IdxSc = Int[Array, ""]


class BufferState(eqx.Module):
    buffer_size: int = eqx.field(static=True)
    index_non_spike_value: int = eqx.field(static=True)
    times: TimeVec
    from_indices: IdxVec
    to_indices: IdxVec


class InternalSpikeBuffer:
    @staticmethod
    def calc_dtype_and_non_spike_value(n_neurons: int) -> Tuple[jnp.dtype, int]:
        if n_neurons <= 255:
            idx_dtype = jnp.uint8
        elif n_neurons <= 2**16:
            idx_dtype = jnp.uint16
        elif n_neurons <= 2**32:
            idx_dtype = jnp.uint32
        else:
            idx_dtype = jnp.uint64
        return idx_dtype, jnp.iinfo(idx_dtype).max

    @staticmethod
    def init(
        buffer_size: int,
        n_neurons: int = 2**32 - 1,
        times: TimeVec | None = None,
        from_indices: IdxVec | None = None,
        to_indices: IdxVec | None = None,
        time_dtype=jnp.float32,
    ) -> BufferState:

        idx_dtype, index_non_spike_value = InternalSpikeBuffer.calc_dtype_and_non_spike_value(n_neurons)

        times_inf = jnp.full(buffer_size + 1, jnp.inf, dtype=time_dtype)
        indices_inf = jnp.full(buffer_size + 1, index_non_spike_value, dtype=idx_dtype)

        if times is None:
            return BufferState(
                buffer_size,
                index_non_spike_value,
                times_inf[:-1],
                indices_inf[:-1],
                indices_inf[:-1],
            )

        state = BufferState(
            buffer_size,
            index_non_spike_value,
            times_inf,
            indices_inf,
            indices_inf,
        )

        times = jnp.asarray(times, dtype=time_dtype)
        from_indices = jnp.asarray(from_indices, dtype=idx_dtype)
        to_indices = jnp.asarray(to_indices, dtype=idx_dtype)
        return InternalSpikeBuffer.add_multiple(state, times, from_indices, to_indices)

    @staticmethod
    def add(state: BufferState, t: TimeSc, from_idx: IdxSc, to_idx: IdxSc) -> BufferState:
        insert_idx = jnp.searchsorted(state.times, t)
        new_times = jnp.insert(state.times, insert_idx, t)[:-1]
        new_from = jnp.insert(state.from_indices, insert_idx, from_idx)[:-1]
        new_to = jnp.insert(state.to_indices, insert_idx, to_idx)[:-1]
        return BufferState(state.buffer_size, state.index_non_spike_value, new_times, new_from, new_to)

    @staticmethod
    def pop(state: BufferState) -> Tuple[TimeSc, IdxSc, IdxSc, BufferState]:
        t = state.times[0]
        from_idx = state.from_indices[0]
        to_idx = state.to_indices[0]
        new_times = jnp.concatenate([state.times[1:], jnp.array([jnp.inf], dtype=state.times.dtype)])
        new_from = jnp.concatenate(
            [state.from_indices[1:], jnp.array([state.index_non_spike_value], dtype=state.from_indices.dtype)]
        )
        new_to = jnp.concatenate(
            [state.to_indices[1:], jnp.array([state.index_non_spike_value], dtype=state.to_indices.dtype)]
        )
        return t, from_idx, to_idx, BufferState(
            state.buffer_size,
            state.index_non_spike_value,
            new_times,
            new_from,
            new_to,
        )

    @staticmethod
    def add_multiple(state: BufferState, new_times: TimeVec, new_from: IdxVec, new_to: IdxVec) -> BufferState:
        all_t = jnp.concatenate([state.times, new_times])
        all_f = jnp.concatenate([state.from_indices, new_from])
        all_to = jnp.concatenate([state.to_indices, new_to])

        # add one additional element for overflow check
        neg, idx = jax.lax.top_k(-all_t, k=state.buffer_size + 1)
        neg = eqx.error_if(neg, neg[-1] != -jnp.inf, "Buffer overflow! Try to increase buffer capacity.")

        # remove check element
        neg = neg[:-1]
        idx = idx[:-1]

        new_times = -neg
        new_from = all_f[idx]
        new_to = all_to[idx]
        return BufferState(state.buffer_size, state.index_non_spike_value, new_times, new_from, new_to)

    @staticmethod
    def peek(state: BufferState) -> Tuple[TimeSc, IdxSc, IdxSc]:
        return state.times[0], state.from_indices[0], state.to_indices[0]

    @staticmethod
    def is_full(state: BufferState) -> Bool[Array, ""]:
        return state.times[-1] != jnp.inf

    @staticmethod
    def is_empty(state: BufferState) -> Bool[Array, ""]:
        return state.times[0] == jnp.inf

    @staticmethod
    def size(state: BufferState) -> Int[Array, ""]:
        return jnp.sum(state.times != jnp.inf)

    @staticmethod
    def capacity(state: BufferState) -> int:
        return state.buffer_size

    @staticmethod
    def get_times(state: BufferState) -> TimeVec:
        return state.times[state.times != jnp.inf]

    @staticmethod
    def get_from_indices(state: BufferState) -> IdxVec:
        return state.from_indices[state.from_indices != state.index_non_spike_value]

    @staticmethod
    def get_to_indices(state: BufferState) -> IdxVec:
        return state.to_indices[state.to_indices != state.index_non_spike_value]

    @staticmethod
    def copy(state: BufferState) -> BufferState:
        return BufferState(
            state.buffer_size,
            state.index_non_spike_value,
            state.times,
            state.from_indices,
            state.to_indices,
        )

    @staticmethod
    def to_str(state: BufferState, display_empty: bool = True) -> str:
        time_col_width = max(
            len(f"{jnp.max(arr):.3f}") if (arr := state.times[state.times != jnp.inf]).size != 0 else 0, 4
        )
        from_col_width = max(
            len(str(jnp.max(arr))) if (
                arr := state.from_indices[state.from_indices != state.index_non_spike_value]).size != 0 else 0, 11
        )
        to_col_width = max(
            len(str(jnp.max(arr))) if (
                arr := state.to_indices[state.to_indices != state.index_non_spike_value]).size != 0 else 0, 9
        )
        header = (f"{'Time'.ljust(time_col_width)} | {'From Neuron'.ljust(from_col_width)}|"
                  f"{'To Neuron'.ljust(to_col_width)}")
        separator = '-' * (time_col_width + from_col_width + to_col_width + 7)
        rows = [
            (f"{('-' if t == jnp.inf else f'{t:.3f}').ljust(time_col_width)} | "
             f"{('-' if x == state.index_non_spike_value else str(x)).ljust(from_col_width)} | "
             f"{('-' if y == state.index_non_spike_value else str(y)).ljust(to_col_width)}")
            for t, x, y in zip(state.times, state.from_indices, state.to_indices)
            if display_empty or (t != jnp.inf and x != state.index_non_spike_value and y != state.index_non_spike_value)
        ]
        table = "\n".join([header, separator] + rows)
        return f"{table}\n{'-' * len(separator)}"


class ExternalBufferState(eqx.Module):
    buffer_size: int = eqx.field(static=True)
    index_non_spike_value: int = eqx.field(static=True)

    current_index: Int[Array, ""]
    times: TimeVec
    from_indices: IdxVec
    to_indices: IdxVec


class ExternalSpikeBuffer:
    @staticmethod
    def init(
        times: TimeVec,
        from_indices: IdxVec,
        to_indices: IdxVec,
        n_neurons: int = 2**32 - 1,
        time_dtype=jnp.float32,
    ) -> ExternalBufferState:
        idx_dtype, index_non_spike_value = InternalSpikeBuffer.calc_dtype_and_non_spike_value(n_neurons)

        times = jnp.asarray(times, dtype=time_dtype)
        from_indices = jnp.asarray(from_indices, dtype=idx_dtype)
        to_indices = jnp.asarray(to_indices, dtype=idx_dtype)

        buffer_size = times.shape[0]

        if buffer_size == 0:
            return ExternalBufferState(
                buffer_size=0,
                index_non_spike_value=index_non_spike_value,
                current_index=jnp.array(0, dtype=jnp.int32),
                times=jnp.asarray([], dtype=time_dtype),
                from_indices=jnp.asarray([], dtype=idx_dtype),
                to_indices=jnp.asarray([], dtype=idx_dtype),
            )

        perm = jnp.argsort(times)
        times = times[perm]
        from_indices = from_indices[perm]
        to_indices = to_indices[perm]

        return ExternalBufferState(
            buffer_size=buffer_size,
            index_non_spike_value=index_non_spike_value,
            current_index=jnp.array(0, dtype=jnp.int32),
            times=times,
            from_indices=from_indices,
            to_indices=to_indices,
        )

    @staticmethod
    def peek(state: ExternalBufferState) -> Tuple[TimeSc, IdxSc, IdxSc]:
        valid = state.current_index < state.buffer_size

        inf_time = jnp.array(jnp.inf, dtype=state.times.dtype)
        ns_val = jnp.array(state.index_non_spike_value, dtype=state.from_indices.dtype)

        if state.buffer_size == 0:
            return inf_time, ns_val, ns_val

        t = jnp.where(valid, state.times[state.current_index], inf_time)
        from_idx = jnp.where(valid, state.from_indices[state.current_index], ns_val)
        to_idx = jnp.where(valid, state.to_indices[state.current_index], ns_val)
        return t, from_idx, to_idx

    @staticmethod
    def pop(state: ExternalBufferState) -> Tuple[TimeSc, IdxSc, IdxSc, ExternalBufferState]:
        t, from_idx, to_idx = ExternalSpikeBuffer.peek(state)
        new_index = jnp.minimum(state.current_index + 1, state.buffer_size)
        new_state = ExternalBufferState(
            buffer_size=state.buffer_size,
            index_non_spike_value=state.index_non_spike_value,
            current_index=new_index,
            times=state.times,
            from_indices=state.from_indices,
            to_indices=state.to_indices,
        )
        return t, from_idx, to_idx, new_state

    @staticmethod
    def is_empty(state: ExternalBufferState) -> Bool[Array, ""]:
        if state.buffer_size == 0:
            return jnp.array(True)

        no_more = state.current_index >= state.buffer_size
        current_time = jnp.where(
            no_more,
            jnp.array(jnp.inf, dtype=state.times.dtype),
            state.times[state.current_index],
        )
        return jnp.logical_or(no_more, jnp.isinf(current_time))

    @staticmethod
    def size(state: ExternalBufferState) -> Int[Array, ""]:
        return state.buffer_size - jnp.minimum(state.current_index, state.buffer_size)

    @staticmethod
    def capacity(state: ExternalBufferState) -> int:
        return state.buffer_size

    @staticmethod
    def get_times(state: ExternalBufferState) -> TimeVec:
        return state.times[state.current_index:]

    @staticmethod
    def get_from_indices(state: ExternalBufferState) -> IdxVec:
        return state.from_indices[state.current_index:]

    @staticmethod
    def get_to_indices(state: ExternalBufferState) -> IdxVec:
        return state.to_indices[state.current_index:]

    @staticmethod
    def copy(state: ExternalBufferState) -> ExternalBufferState:
        return ExternalBufferState(
            state.buffer_size,
            state.index_non_spike_value,
            state.current_index,
            state.times,
            state.from_indices,
            state.to_indices,
        )


class SpikeBufferState(eqx.Module):
    external_state: ExternalBufferState
    internal_state: BufferState
    index_non_spike_value: int = eqx.field(static=True)


class SpikeBuffer:
    @staticmethod
    def calc_dtype_and_non_spike_value(n_neurons: int) -> Tuple[jnp.dtype, int]:
        return InternalSpikeBuffer.calc_dtype_and_non_spike_value(n_neurons)

    @staticmethod
    def init(
        buffer_size: int,
        n_neurons: int = 2**32 - 1,
        times: TimeVec | None = None,
        from_indices: IdxVec | None = None,
        to_indices: IdxVec | None = None,
        time_dtype=jnp.float32,
    ) -> SpikeBufferState:
        idx_dtype, index_non_spike_value = SpikeBuffer.calc_dtype_and_non_spike_value(n_neurons)

        if times is None:
            external_state = ExternalSpikeBuffer.init(
                jnp.asarray([], dtype=time_dtype),
                jnp.asarray([], dtype=idx_dtype),
                jnp.asarray([], dtype=idx_dtype),
                n_neurons=n_neurons,
                time_dtype=time_dtype,
            )
        else:
            external_state = ExternalSpikeBuffer.init(
                times,
                from_indices,
                to_indices,
                n_neurons=n_neurons,
                time_dtype=time_dtype,
            )

        internal_state = InternalSpikeBuffer.init(
            buffer_size=buffer_size,
            n_neurons=n_neurons,
            times=None,
            from_indices=None,
            to_indices=None,
            time_dtype=time_dtype,
        )

        return SpikeBufferState(
            external_state=external_state,
            internal_state=internal_state,
            index_non_spike_value=index_non_spike_value,
        )

    @staticmethod
    def add(state: SpikeBufferState, t: TimeSc, from_idx: IdxSc, to_idx: IdxSc) -> SpikeBufferState:
        new_internal = InternalSpikeBuffer.add(state.internal_state, t, from_idx, to_idx)
        return SpikeBufferState(
            external_state=state.external_state,
            internal_state=new_internal,
            index_non_spike_value=state.index_non_spike_value,
        )

    @staticmethod
    def add_multiple(
        state: SpikeBufferState,
        new_times: TimeVec,
        new_from: IdxVec,
        new_to: IdxVec,
    ) -> SpikeBufferState:
        new_internal = InternalSpikeBuffer.add_multiple(state.internal_state, new_times, new_from, new_to)
        return SpikeBufferState(
            external_state=state.external_state,
            internal_state=new_internal,
            index_non_spike_value=state.index_non_spike_value,
        )

    @staticmethod
    def peek(state: SpikeBufferState) -> Tuple[TimeSc, IdxSc, IdxSc]:
        t_ext, f_ext, to_ext = ExternalSpikeBuffer.peek(state.external_state)
        t_int, f_int, to_int = InternalSpikeBuffer.peek(state.internal_state)

        use_external = t_ext <= t_int

        t = jnp.where(use_external, t_ext, t_int)
        f = jnp.where(use_external, f_ext, f_int)
        to = jnp.where(use_external, to_ext, to_int)
        return t, f, to

    @staticmethod
    def pop(state: SpikeBufferState) -> Tuple[TimeSc, IdxSc, IdxSc, SpikeBufferState]:
        t_ext, _, _ = ExternalSpikeBuffer.peek(state.external_state)
        t_int, _, _ = InternalSpikeBuffer.peek(state.internal_state)

        use_external = t_ext <= t_int

        def pop_external(s: SpikeBufferState):
            t, f, to, new_ext = ExternalSpikeBuffer.pop(s.external_state)
            new_state = SpikeBufferState(
                external_state=new_ext,
                internal_state=s.internal_state,
                index_non_spike_value=s.index_non_spike_value,
            )
            return t, f, to, new_state

        def pop_internal(s: SpikeBufferState):
            t, f, to, new_int = InternalSpikeBuffer.pop(s.internal_state)
            new_state = SpikeBufferState(
                external_state=s.external_state,
                internal_state=new_int,
                index_non_spike_value=s.index_non_spike_value,
            )
            return t, f, to, new_state

        return jax.lax.cond(use_external, pop_external, pop_internal, state)

    @staticmethod
    def is_full(state: SpikeBufferState) -> Bool[Array, ""]:
        return InternalSpikeBuffer.is_full(state.internal_state)

    @staticmethod
    def is_empty(state: SpikeBufferState) -> Bool[Array, ""]:
        return jnp.logical_and(
            ExternalSpikeBuffer.is_empty(state.external_state),
            InternalSpikeBuffer.is_empty(state.internal_state),
        )

    @staticmethod
    def size(state: SpikeBufferState) -> Int[Array, ""]:
        return ExternalSpikeBuffer.size(state.external_state) + InternalSpikeBuffer.size(state.internal_state)

    @staticmethod
    def capacity(state: SpikeBufferState) -> int:
        return ExternalSpikeBuffer.capacity(state.external_state) + InternalSpikeBuffer.capacity(state.internal_state)

    @staticmethod
    def _gather_all_sorted(state: SpikeBufferState) -> Tuple[TimeVec, IdxVec, IdxVec]:
        t_ext = ExternalSpikeBuffer.get_times(state.external_state)
        f_ext = ExternalSpikeBuffer.get_from_indices(state.external_state)
        to_ext = ExternalSpikeBuffer.get_to_indices(state.external_state)

        t_int = InternalSpikeBuffer.get_times(state.internal_state)
        f_int = InternalSpikeBuffer.get_from_indices(state.internal_state)
        to_int = InternalSpikeBuffer.get_to_indices(state.internal_state)

        all_t = jnp.concatenate([t_ext, t_int])
        all_f = jnp.concatenate([f_ext, f_int])
        all_to = jnp.concatenate([to_ext, to_int])

        if all_t.size == 0:
            return all_t, all_f, all_to

        perm = jnp.argsort(all_t)
        return all_t[perm], all_f[perm], all_to[perm]

    @staticmethod
    def get_times(state: SpikeBufferState) -> TimeVec:
        all_t, _, _ = SpikeBuffer._gather_all_sorted(state)
        return all_t

    @staticmethod
    def get_from_indices(state: SpikeBufferState) -> IdxVec:
        _, all_f, _ = SpikeBuffer._gather_all_sorted(state)
        return all_f

    @staticmethod
    def get_to_indices(state: SpikeBufferState) -> IdxVec:
        _, _, all_to = SpikeBuffer._gather_all_sorted(state)
        return all_to

    @staticmethod
    def copy(state: SpikeBufferState) -> SpikeBufferState:
        ext = ExternalSpikeBuffer.copy(state.external_state)
        inte = InternalSpikeBuffer.copy(state.internal_state)
        return SpikeBufferState(
            external_state=ext,
            internal_state=inte,
            index_non_spike_value=state.index_non_spike_value,
        )

    @staticmethod
    def to_str(state: SpikeBufferState, display_empty: bool = True) -> str:
        times, from_indices, to_indices = SpikeBuffer._gather_all_sorted(state)

        if times.size == 0 and not display_empty:
            return "<empty spike buffer>"

        time_col_width = max(
            len(f"{jnp.max(times):.3f}") if times.size != 0 else 0,
            4,
        )
        from_col_width = max(
            len(str(jnp.max(from_indices))) if from_indices.size != 0 else 0,
            11,
        )
        to_col_width = max(
            len(str(jnp.max(to_indices))) if to_indices.size != 0 else 0,
            9,
        )

        header = (
            f"{'Time'.ljust(time_col_width)} | "
            f"{'From Neuron'.ljust(from_col_width)}|"
            f"{'To Neuron'.ljust(to_col_width)}"
        )
        separator = '-' * (time_col_width + from_col_width + to_col_width + 7)

        rows = [
            (
                f"{f'{t:.3f}'.ljust(time_col_width)} | "
                f"{str(x).ljust(from_col_width)} | "
                f"{str(y).ljust(to_col_width)}"
            )
            for t, x, y in zip(times, from_indices, to_indices)
        ]
        table = "\n".join([header, separator] + rows)
        return f"{table}\n{'-' * len(separator)}"
