import jax
import jax.numpy as jnp


def compute_kinetic_energy(model, x):
    d2psi_dx2 = jax.grad(lambda x_val: jax.grad(model)(x_val))(x)
    return -0.5 * d2psi_dx2


def double_well_potential(x):
    # Non-linear potential with minima at x = -2 and x = 2
    return 0.1 * (x ** 2 - 4.0) ** 2


def variational_loss(model, x_grid, potential_fn, ground_state_model=None, penalty_weight=100.0):
    vmap_psi = jax.vmap(model)
    vmap_kinetic = jax.vmap(lambda x: compute_kinetic_energy(model, x))
    vmap_potential = jax.vmap(potential_fn)

    psi_vals = vmap_psi(x_grid)
    kinetic_vals = vmap_kinetic(x_grid)
    potential_vals = vmap_potential(x_grid)

    H_psi = kinetic_vals + (potential_vals * psi_vals)

    expected_energy = jnp.trapezoid(psi_vals * H_psi, x=x_grid)
    norm_factor = jnp.trapezoid(psi_vals ** 2, x=x_grid)

    base_energy = expected_energy / norm_factor

    # Gram-Schmidt Orthogonalization Penalty
    if ground_state_model is not None:
        vmap_ground = jax.vmap(ground_state_model)
        ground_vals = vmap_ground(x_grid)
        # Inner product <psi_theta | psi_0>
        overlap = jnp.trapezoid(psi_vals * ground_vals, x=x_grid)
        # Square the overlap to create a massive positive penalty
        overlap_penalty = penalty_weight * (overlap ** 2)
        return base_energy + overlap_penalty

    return base_energy