import jax
import jax.numpy as jnp


# 1. Autograd for Spatiotemporal Derivatives
def compute_pde_residuals(model, t, x, potential_fn):
    # Helper functions to isolate u and v for clean differentiation
    def get_u(t_val, x_val): return model(t_val, x_val)[0]

    def get_v(t_val, x_val): return model(t_val, x_val)[1]

    # Time derivatives (1st order)
    u_t = jax.grad(get_u, argnums=0)(t, x)
    v_t = jax.grad(get_v, argnums=0)(t, x)

    # Spatial derivatives (2nd order)
    u_xx = jax.grad(lambda t_v, x_v: jax.grad(get_u, argnums=1)(t_v, x_v), argnums=1)(t, x)
    v_xx = jax.grad(lambda t_v, x_v: jax.grad(get_v, argnums=1)(t_v, x_v), argnums=1)(t, x)

    # Evaluate potential at x
    V = potential_fn(x)
    u, v = model(t, x)

    # The Coupled TDSE Residuals (Target: 0)
    f_u = v_t - 0.5 * u_xx + V * u
    f_v = u_t + 0.5 * v_xx - V * v

    return f_u, f_v


# 2. Initial Condition: A Moving Gaussian Wavepacket
def initial_wavepacket(x, x0=-3.0, p0=2.0, sigma=0.5):
    # A standard Gaussian packet starting at x0, with initial momentum p0
    envelope = (1.0 / (jnp.pi * sigma ** 2)) ** 0.25 * jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

    # Euler's formula for the momentum phase: e^(i * p0 * x) = cos(p0*x) + i*sin(p0*x)
    u_init = envelope * jnp.cos(p0 * x)
    v_init = envelope * jnp.sin(p0 * x)

    return u_init, v_init


# 3. The Unified Loss Function
def physics_loss(model, t_collocation, x_collocation, x_initial, potential_fn):
    # --- PART A: PDE Residual Loss (Physics in Spacetime) ---
    vmap_residuals = jax.vmap(lambda t, x: compute_pde_residuals(model, t, x, potential_fn))
    f_u, f_v = vmap_residuals(t_collocation, x_collocation)
    loss_pde = jnp.mean(f_u ** 2 + f_v ** 2)

    # --- PART B: Initial Condition Loss (t=0) ---
    vmap_model_init = jax.vmap(lambda x: model(0.0, x))
    u_pred, v_pred = vmap_model_init(x_initial)

    vmap_true_init = jax.vmap(initial_wavepacket)
    u_true, v_true = vmap_true_init(x_initial)

    loss_ic = jnp.mean((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)

    # Total loss: The network must satisfy the initial shape AND flow correctly through time
    return loss_pde + (10.0 * loss_ic)  # heavily weight the initial condition