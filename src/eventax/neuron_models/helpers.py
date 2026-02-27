import jax.numpy as jnp
import jax


def clip_with_identity_grad(x, max_val):
    """
    Clamp x from above in the forward pass (x <= max_val), but keep d/dx = 1
    even when the clamp is active.

    This is used as a numerical safety net around spike events: if the ODE
    solver ever steps past threshold and misses an event because it detected
    another actually later event erlier, the state of the event that was not detected
    is projected back just below threshold so it can likely trigger a spike in the next step.

    The identity gradient is important for state-based losses: even when we
    correct the state numerically, we still want gradients to flow through
    that state as if it had not been clamped, so the loss can influence
    earlier dynamics.
    """
    excess = jnp.maximum(0.0, x - max_val)
    return x - jax.lax.stop_gradient(excess)
