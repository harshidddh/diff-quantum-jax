import jax
import jax.numpy as jnp


# The "Experimental Data" we observed in the lab
def experimental_wavefunction(x):
    return (1.0 / jnp.pi) ** 0.25 * jnp.exp(-0.5 * x ** 2)


def inverse_residual_loss(potential_model, x_grid, E_measured=0.5):
    # JAX allows us to get the exact analytical derivatives of our experimental data
    def psi(x): return experimental_wavefunction(x)

    def d2psi_dx2(x): return jax.grad(lambda x_val: jax.grad(psi)(x_val))(x)

    vmap_psi = jax.vmap(psi)
    vmap_d2psi = jax.vmap(d2psi_dx2)
    vmap_V = jax.vmap(potential_model)

    psi_vals = vmap_psi(x_grid)
    d2psi_vals = vmap_d2psi(x_grid)

    # The neural network predicts the potential at these coordinates
    V_pred = vmap_V(x_grid)

    # TISE Rearranged: The physical residual that must equal 0
    # f = -0.5 * psi'' + V_pred * psi - E * psi
    residual = -0.5 * d2psi_vals + V_pred * psi_vals - E_measured * psi_vals

    # Mean Squared Error of the physical violation
    return jnp.mean(residual ** 2)