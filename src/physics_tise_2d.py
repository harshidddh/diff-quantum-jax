import jax
import jax.numpy as jnp


def compute_2d_kinetic_energy(model, x, y):
    # d^2(psi)/dx^2
    d2psi_dx2 = jax.grad(lambda x_val, y_val: jax.grad(model, argnums=0)(x_val, y_val), argnums=0)(x, y)
    # d^2(psi)/dy^2
    d2psi_dy2 = jax.grad(lambda x_val, y_val: jax.grad(model, argnums=1)(x_val, y_val), argnums=1)(x, y)

    # 2D Laplacian
    return -0.5 * (d2psi_dx2 + d2psi_dy2)


def variational_loss_2d(model, x_grid, y_grid):
    # Create a 2D grid
    X, Y = jnp.meshgrid(x_grid, y_grid)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    vmap_psi = jax.vmap(model)
    vmap_kinetic = jax.vmap(compute_2d_kinetic_energy, in_axes=(None, 0, 0))

    psi_vals = vmap_psi(x_flat, y_flat)
    kinetic_vals = vmap_kinetic(model, x_flat, y_flat)

    # For an infinite empty box, V(x,y) = 0 inside.
    # The boundary conditions are already handled by the model's envelope!
    H_psi = kinetic_vals

    # Expectation value <psi|H|psi> using 2D sum (approximation of 2D integral)
    expected_energy = jnp.sum(psi_vals * H_psi)
    norm_factor = jnp.sum(psi_vals ** 2)

    return expected_energy / norm_factor