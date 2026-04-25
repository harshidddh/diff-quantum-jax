import jax
import jax.numpy as jnp


def compute_kinetic_energy(model, x):
    d2psi_dx2 = jax.grad(lambda x_val: jax.grad(model)(x_val))(x)
    return -0.5 * d2psi_dx2


def double_well_potential(x):
    return 0.1 * (x ** 2 - 4.0) ** 2


def variational_loss(model, x_grid, potential_fn):
    vmap_psi = jax.vmap(model)
    vmap_kinetic = jax.vmap(lambda x: compute_kinetic_energy(model, x))
    vmap_potential = jax.vmap(potential_fn)

    psi_vals = vmap_psi(x_grid)
    kinetic_vals = vmap_kinetic(x_grid)
    potential_vals = vmap_potential(x_grid)

    H_psi = kinetic_vals + (potential_vals * psi_vals)

    expected_energy = jnp.trapezoid(psi_vals * H_psi, x=x_grid)
    norm_factor = jnp.trapezoid(psi_vals ** 2, x=x_grid)

    # Pure Variational Energy. No penalties needed!
    return expected_energy / norm_factor