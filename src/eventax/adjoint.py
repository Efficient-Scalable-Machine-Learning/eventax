import functools as ft
import warnings
from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from diffrax._heuristics import is_sde, is_unsafe_sde
from diffrax._saveat import save_y, SaveAt, SubSaveAt
from diffrax._solver import (
    AbstractItoSolver,
    AbstractStratonovichSolver,
)
from diffrax._term import AbstractTerm, AdjointTerm
from diffrax._adjoint import AbstractAdjoint

ω = cast(Callable, ω)


def _is_none(x):
    return x is None


def _is_subsaveat(x: Any) -> bool:
    return isinstance(x, SubSaveAt)


def _nondiff_solver_controller_state(
    adjoint, init_state, passed_solver_state, passed_controller_state
):
    if passed_solver_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then `solver_state`"
        )
        solver_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        solver_fn = lax.stop_gradient
    if passed_controller_state:
        name = (
            f"When using `adjoint={adjoint.__class__.__name__}()`, then "
            "`controller_state`"
        )
        controller_fn = ft.partial(
            eqxi.nondifferentiable,
            name=name,
        )
    else:
        controller_fn = lax.stop_gradient
    init_state = eqx.tree_at(
        lambda s: s.solver_state,
        init_state,
        replace_fn=solver_fn,
        is_leaf=_is_none,
    )
    init_state = eqx.tree_at(
        lambda s: s.controller_state,
        init_state,
        replace_fn=controller_fn,
        is_leaf=_is_none,
    )
    return init_state


def _only_transpose_ys(final_state):
    from diffrax._integrate import SaveState

    def is_save_state(x): return isinstance(x, SaveState)

    def get_ys(_final_state):
        return [
            s.ys
            for s in jtu.tree_leaves(_final_state.save_state, is_leaf=is_save_state)
        ]

    def get_ts(_final_state):
        return [
            s.ts
            for s in jtu.tree_leaves(_final_state.save_state, is_leaf=is_save_state)
        ]

    ys = get_ys(final_state)
    ts = get_ts(final_state)

    named_nondiff_entries = (
        "y",
        "tprev",
        "tnext",
        "solver_state",
        "controller_state",
        "dense_ts",
        "dense_infos",
    )
    named_nondiff_values = tuple(
        eqxi.nondifferentiable_backward(getattr(final_state, k), name=k, symbolic=False)
        for k in named_nondiff_entries
    )

    final_state = eqxi.nondifferentiable_backward(final_state, symbolic=False)

    def get_named_nondiff_entries(s): return tuple(
        getattr(s, k) for k in named_nondiff_entries
    )
    final_state = eqx.tree_at(
        get_named_nondiff_entries, final_state, named_nondiff_values, is_leaf=_is_none
    )

    final_state = eqx.tree_at(get_ys, final_state, ys)
    final_state = eqx.tree_at(get_ts, final_state, ts)
    return final_state


_inner_loop = jax.named_call(eqxi.while_loop, name="inner-loop")
_outer_loop = jax.named_call(eqxi.while_loop, name="outer-loop")


@eqx.filter_custom_vjp
def _loop_backsolve(y__args__terms__t0__t1, *, self, throw, init_state, **kwargs):
    del throw
    y, args, terms, t0, t1 = y__args__terms__t0__t1
    init_state = eqx.tree_at(lambda s: s.y, init_state, y)
    del y
    return self._loop(
        args=args,
        terms=terms,
        init_state=init_state,
        inner_while_loop=ft.partial(_inner_loop, kind="lax"),
        outer_while_loop=ft.partial(_outer_loop, kind="lax"),
        t0=t0,
        t1=t1,
        **kwargs,
    )


@_loop_backsolve.def_fwd
def _loop_backsolve_fwd(perturbed, y__args__terms__t0__t1, **kwargs):
    del perturbed
    final_state, aux_stats = _loop_backsolve(y__args__terms__t0__t1, **kwargs)
    # Note that `final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `EventPropAdjoint` that it have trivial structure.
    ts = final_state.save_state.ts
    ys = final_state.save_state.ys

    event_mask = final_state.event_mask
    event_tprev = final_state.event_tprev
    event_tnext = final_state.event_tnext
    event_dense_info = final_state.event_dense_info
    event_values = final_state.event_values

    return (final_state, aux_stats), (ts, ys, event_mask, event_tprev, event_tnext, event_dense_info, event_values)


def _materialise_none(y, grad_y):
    if grad_y is None and eqx.is_inexact_array(y):
        return jnp.zeros_like(y)
    else:
        return grad_y


@_loop_backsolve.def_bwd
def _loop_backsolve_bwd(
    residuals,
    grad_final_state__aux_stats,
    perturbed,
    y__args__terms__t0__t1,
    *,
    self,
    solver,
    stepsize_controller,
    event,
    saveat,
    dt0,
    max_steps,
    throw,
    init_state,
    progress_meter,
):
    #
    # Unpack our various arguments. Delete a lot of things just to make sure we're not
    # using them later.
    #

    del perturbed, init_state, progress_meter
    ts, ys, event_mask, event_tprev, event_tnext, event_dense_info, event_values = residuals
    del residuals
    grad_final_state, _ = grad_final_state__aux_stats
    # Note that `grad_final_state.save_state` has type `PyTree[SaveState]`; here we are
    # relying on the guard in `EventPropAdjoint` that it have trivial structure.
    grad_ys = grad_final_state.save_state.ys
    grad_ts = grad_final_state.save_state.ts
    # We take the simple way out and don't try to handle symbolic zeros.
    grad_ys = jtu.tree_map(_materialise_none, ys, grad_ys)
    grad_ts = jtu.tree_map(_materialise_none, ts, grad_ts)
    del grad_final_state, grad_final_state__aux_stats
    y, args, terms, t0, t1 = y__args__terms__t0__t1
    del y__args__terms__t0__t1
    diff_args = eqx.filter(args, eqx.is_inexact_array)
    diff_terms = eqx.filter(terms, eqx.is_inexact_array)
    zeros_like_y = jtu.tree_map(jnp.zeros_like, y)
    zeros_like_diff_args = jtu.tree_map(jnp.zeros_like, diff_args)
    zeros_like_diff_terms = jtu.tree_map(jnp.zeros_like, diff_terms)

    # TODO: have this look inside MultiTerms? Need to think about the math. i.e.:
    # is_leaf=lambda x: isinstance(x, AbstractTerm) and not isinstance(x, MultiTerm)
    adjoint_terms = jtu.tree_map(
        AdjointTerm, terms, is_leaf=lambda x: isinstance(x, AbstractTerm)
    )
    diffeqsolve = self._diffeqsolve
    kwargs = dict(
        args=args,
        adjoint=self,
        solver=solver,
        stepsize_controller=stepsize_controller,
        terms=adjoint_terms,
        dt0=None if dt0 is None else -dt0,
        max_steps=max_steps,
        throw=throw,
    )
    kwargs.update(self.kwargs)

    # Note that `saveat.subs` has type `PyTree[SubSaveAt]`. Here we use the assumption
    # (checked in `EventPropAdjoint`) that it has trivial pytree structure.
    saveat_t0 = saveat.subs.t0

    if event is not None and event_mask is not None:
        def _event_contribution():
            """Compute the adjoint jump from the event."""
            # Get the time and state values of the event
            t_event = ω(ts)[-1].ω
            y_event = ω(ys)[-1].ω

            dL_dt_star = ω(grad_ts)[-1].ω
            lambda_plus = ω(grad_ys)[-1].ω  # λ(t*⁺) = dL/dy(t*)

            f_val = terms.vf(t_event, y_event, args)

            def _eval_cond(_y, _t):
                return event.cond_fn(
                    _t, _y, args,
                    terms=terms,
                    solver=solver,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                    saveat=saveat,
                    stepsize_controller=stepsize_controller,
                    max_steps=max_steps,
                )
            _, vjp_fun = eqx.filter_vjp(_eval_cond, y_event, t_event)
            dg_dy, dg_dt = vjp_fun(1.0)

            # Compute ν = -(dL/dt* + λᵀ · f) / (∂g/∂t + ∂g/∂y · f)
            dg_dy_dot_f = jtu.tree_reduce(
                lambda a, b: a + b,
                jtu.tree_map(lambda dy, f: jnp.sum(dy * f), dg_dy, f_val)
            )
            denominator = dg_dt + dg_dy_dot_f

            lambda_dot_f = jtu.tree_reduce(
                lambda a, b: a + b,
                jtu.tree_map(lambda lp, f: jnp.sum(lp * f), lambda_plus, f_val)
            )

            nu = -(dL_dt_star + lambda_dot_f) / (denominator + 1e-12)

            # Δλ = ν · ∂g/∂y
            adjoint_jump = jtu.tree_map(lambda dy: nu * dy, dg_dy)

            # λ(t*⁻) = λ(t*⁺) + Δλ
            lambda_minus = (lambda_plus**ω + adjoint_jump**ω).ω
            return lambda_minus

        def _no_event_contribution():
            """No event occurred, just return the original gradient."""
            return ω(grad_ys)[-1].ω

        # Conditionally compute event contribution
        adjusted_lambda = lax.cond(
            event_mask,
            _event_contribution,
            _no_event_contribution,
        )

        grad_ys = jtu.tree_map(
            lambda g, new_val: g.at[-1].set(new_val),
            grad_ys,
            adjusted_lambda
        )

    del self, solver, stepsize_controller, adjoint_terms, dt0, max_steps, throw
    del saveat
    del diff_args, diff_terms

    #
    # Now run a scan backwards in time, diffeqsolve'ing between each pair of adjacent
    # timestamps.
    #

    def _scan_fun(_state, _vals, first=False):
        _t1, _t0, _y0, _grad_y0 = _vals
        _a0, _solver_state, _controller_state = _state
        _a_y0, _a_diff_args0, _a_diff_term0 = _a0
        _a_y0 = (_a_y0**ω + _grad_y0**ω).ω
        _aug0 = (_y0, _a_y0, _a_diff_args0, _a_diff_term0)

        _sol = diffeqsolve(
            t0=_t0,
            t1=_t1,
            y0=_aug0,
            solver_state=_solver_state,
            controller_state=_controller_state,
            made_jump=not first,  # Adding _grad_y0, above, is a jump.
            saveat=SaveAt(t1=True, solver_state=True, controller_state=True),
            **kwargs,
        )

        def __get(__aug):
            assert __aug.shape[0] == 1
            return __aug[0]

        _aug1 = ω(_sol.ys).call(__get).ω
        _, _a_y1, _a_diff_args1, _a_diff_term1 = _aug1
        _a1 = (_a_y1, _a_diff_args1, _a_diff_term1)
        _solver_state = _sol.solver_state
        _controller_state = _sol.controller_state

        return (_a1, _solver_state, _controller_state), None

    state = ((zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms), None, None)
    del zeros_like_y, zeros_like_diff_args, zeros_like_diff_terms

    # We always start backpropagating from `ts[-1]`.
    # We always finish backpropagating at `t0`.
    #
    # We may or may not have included `t0` in `ts`. (Depending on the value of
    # SaveaAt(t0=...) on the forward pass.)
    #
    # For some of these options, we run _scan_fun once outside the loop to get access
    # to solver_state etc. of the correct PyTree structure.
    if saveat_t0:
        if len(ts) > 2:
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                ts[:-2],
                ts[1:-1],
                ω(ys)[1:-1].ω,
                ω(grad_ys)[1:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        elif len(ts) == 1:
            # nothing to do, diffeqsolve is the identity when merely SaveAt(t0=True).
            pass

        else:
            assert len(ts) == 2
            val = (ts[0], ts[1], ω(ys)[1].ω, ω(grad_ys)[1].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1
        a_y1 = (ω(a_y1) + ω(grad_ys)[0]).ω

    else:
        if len(ts) > 1:
            # TODO: fold this `_scan_fun` into the `lax.scan`. This will reduce compile
            # time.
            val0 = (ts[-2], ts[-1], ω(ys)[-1].ω, ω(grad_ys)[-1].ω)
            state, _ = _scan_fun(state, val0, first=True)
            vals = (
                jnp.concatenate([t0[None], ts[:-2]]),
                ts[:-1],
                ω(ys)[:-1].ω,
                ω(grad_ys)[:-1].ω,
            )
            state, _ = lax.scan(_scan_fun, state, vals, reverse=True)

        else:
            assert len(ts) == 1
            val = (t0, ts[0], ω(ys)[0].ω, ω(grad_ys)[0].ω)
            state, _ = _scan_fun(state, val, first=True)

        aug1, _, _ = state
        a_y1, a_diff_args1, a_diff_terms1 = aug1

    # Boundary conditions
    f_t1 = terms.vf(ts[-1], ω(ys)[-1].ω, args)
    grad_t1 = jtu.tree_reduce(
        lambda a, b: a + b,
        jtu.tree_map(lambda a, f: jnp.sum(a * f), ω(grad_ys)[-1].ω, f_t1)
    )

    f_t0 = terms.vf(t0, y, args)  # Vector field at t0
    grad_t0 = -jtu.tree_reduce(
        lambda a, b: a + b,
        jtu.tree_map(lambda a, f: jnp.sum(a * f), a_y1, f_t0)
    )

    return a_y1, a_diff_args1, a_diff_terms1, grad_t0, grad_t1


class EventPropAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time. This is also sometimes known as
    "optimise-then-discretise", the "continuous adjoint method" or simply the "adjoint
    method".

    This will compute gradients with respect to the `terms`, `y0`, `args`, `t0`, and `t1`
    arguments passed to [`diffrax.diffeqsolve`][]. If you attempt to compute gradients with
    respect to anything else (for example arguments passed via closure), then
    a `CustomVJPException` will be raised by JAX. See also
    [this FAQ](../../further_details/faq/#im-getting-a-customvjpexception)
    entry.

    !!! info

        Using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """  # noqa: E501

    kwargs: dict[str, Any]

    def __init__(self, **kwargs):
        """
        **Arguments:**

        - `**kwargs`: The arguments for the [`diffrax.diffeqsolve`][] operations that
            are called on the backward pass. For example use
            ```python
            EventPropAdjoint(solver=Dopri5())
            ```
            to specify a particular solver to use on the backward pass.
        """
        valid_keys = {
            "dt0",
            "solver",
            "stepsize_controller",
            "adjoint",
            "max_steps",
            "throw",
        }
        given_keys = set(kwargs.keys())
        diff_keys = given_keys - valid_keys
        if len(diff_keys) > 0:
            raise ValueError(
                "The following keyword argments are not valid for `EventPropAdjoint`: "
                f"{diff_keys}"
            )
        self.kwargs = kwargs

    def loop(
        self,
        *,
        args,
        terms,
        solver,
        saveat,
        init_state,
        passed_solver_state,
        passed_controller_state,
        event,
        t0,
        t1,
        dt0,
        **kwargs,
    ):
        if jtu.tree_structure(saveat.subs, is_leaf=_is_subsaveat) != jtu.tree_structure(
            0
        ):
            raise NotImplementedError(
                "Cannot use `adjoint=EventPropAdjoint()` with `SaveAt(subs=...)`."
            )
        if saveat.dense or saveat.subs.steps:
            raise NotImplementedError(
                "Cannot use `adjoint=EventPropAdjoint()` with "
                "`saveat=SaveAt(steps=True)` or saveat=SaveAt(dense=True)`."
            )
        if saveat.subs.fn is not save_y:
            raise NotImplementedError(
                "Cannot use `adjoint=EventPropAdjoint()` with `saveat=SaveAt(fn=...)`."
            )
        if is_unsafe_sde(terms):
            raise ValueError(
                "`adjoint=EventPropAdjoint()` does not support `UnsafeBrownianPath`. "
                "Consider using `adjoint=DirectAdjoint()` instead."
            )
        if is_sde(terms):
            if isinstance(solver, AbstractItoSolver):
                raise NotImplementedError(
                    f"`{solver.__class__.__name__}` converges to the Itô solution. "
                    "However `EventPropAdjoint` currently only supports Stratonovich "
                    "SDEs."
                )
            elif not isinstance(solver, AbstractStratonovichSolver):
                warnings.warn(
                    f"{solver.__class__.__name__} is not marked as converging to "
                    "either the Itô or the Stratonovich solution. Note that "
                    "`EventPropAdjoint` will only produce the correct solution for "
                    "Stratonovich SDEs."
                )
        if jtu.tree_structure(solver.term_structure) != jtu.tree_structure(0):
            raise NotImplementedError(
                "`diffrax.EventPropAdjoint` is only compatible with solvers that take "
                "a single term."
            )

        y = init_state.y
        init_state = eqx.tree_at(lambda s: s.y, init_state, object())
        # jax.debug.print("{x}", x=init_state)
        init_state = jax.tree.map(lambda x: lax.stop_gradient(x) if eqx.is_array(x) else x, init_state)
        init_state = _nondiff_solver_controller_state(
            self, init_state, passed_solver_state, passed_controller_state
        )

        final_state, aux_stats = _loop_backsolve(
            (y, args, terms, t0, t1),
            self=self,
            saveat=saveat,
            init_state=init_state,
            solver=solver,
            event=event,
            dt0=dt0,
            **kwargs,
        )
        final_state = _only_transpose_ys(final_state)
        return final_state, aux_stats
