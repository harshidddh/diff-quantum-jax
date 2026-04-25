import jax
import jax.numpy as jnp


def compute_kinetic_energy(model, x):
    # Differentiating the network with respect to x twice
    # 1st derivative: d(psi)/dx
    dpsi_dx = jax.grad(model)(x)
    # 2nd derivative: d^2(psi)/dx^2
    d2psi_dx2 = jax.grad(lambda x_val: jax.grad(model)(x_val))(x)
    return -0.5 * d2psi_dx2


def compute_potential_energy(x):
    # Harmonic Oscillator Potential: V(x) = 1/2 m w^2 x^2 (in natural units)
    return 0.5 * x ** 2


def variational_loss(model, x_grid):
    # Vectorize the operations over the spatial grid
    vmap_psi = jax.vmap(model)
    vmap_kinetic = jax.vmap(lambda x: compute_kinetic_energy(model, x))
    vmap_potential = jax.vmap(compute_potential_energy)

    psi_vals = vmap_psi(x_grid)
    kinetic_vals = vmap_kinetic(x_grid)
    potential_vals = vmap_potential(x_grid)

    # Calculate H|psi>
    H_psi = kinetic_vals + (potential_vals * psi_vals)

    # Expectation value: <psi|H|psi>
    expected_energy = jnp.trapezoid(psi_vals * H_psi, x=x_grid)

    # Normalization factor: <psi|psi>
    norm_factor = jnp.trapezoidz(psi_vals ** 2, x=x_grid)

    # The variational energy E (this is what the optimizer will minimize)
    return expected_energy / norm_factor