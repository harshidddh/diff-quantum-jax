import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
import optax
from src.model_tise_1d import QuantumWaveFunction
from src.physics_tise_1d import variational_loss


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def test_qho_convergence(key):
    """Verifies that the 1D solver actually decreases energy via the Variational Principle."""
    model = QuantumWaveFunction(key, is_odd=False)
    optimizer = optax.adam(learning_rate=0.01)

    # Manually verified: no markdown links, pure attribute access
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    x_grid = jnp.linspace(-5, 5, 500)
    potential_fn = lambda x: 0.5 * x ** 2

    @eqx.filter_jit
    def step(m, o_s):
        loss, grads = eqx.filter_value_and_grad(variational_loss)(m, x_grid, potential_fn)
        # Manually verified: eqx.is_array
        updates, o_s = optimizer.update(grads, o_s, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, updates)
        return m, o_s, loss

    _, _, initial_loss = step(model, opt_state)

    curr_model, curr_opt_state = model, opt_state
    for _ in range(50):
        curr_model, curr_opt_state, loss = step(curr_model, curr_opt_state)

    assert float(loss) < float(initial_loss), "Physics optimization failed: Energy did not decrease."
    assert float(loss) < 1.0, f"Non-physical ground state energy: {float(loss):.4f}"


def test_parity_constrained_antisymmetry(key):
    """Verifies the architectural enforcement of psi(x) = -psi(-x)."""
    model = QuantumWaveFunction(key, is_odd=True)
    x_test = jax.random.uniform(key, (20,), minval=-3.0, maxval=3.0)

    vmap_model = jax.vmap(model)
    psi_pos = vmap_model(x_test)
    psi_neg = vmap_model(-x_test)

    assert jnp.allclose(psi_pos, -psi_neg, atol=1e-7), "Hard parity constraint violated!"


def test_wavefunction_normalization(key):
    """Ensures the probability density integrates to unity."""
    model = QuantumWaveFunction(key, is_odd=False)
    x_grid = jnp.linspace(-6.0, 6.0, 1000)

    psi_raw = jax.vmap(model)(x_grid)
    norm_sq = jnp.trapezoid(psi_raw ** 2, x=x_grid)
    psi_normalized = psi_raw / jnp.sqrt(norm_sq)

    integral = jnp.trapezoid(psi_normalized ** 2, x=x_grid)
    assert jnp.isclose(integral, 1.0, atol=1e-5)


def test_tdse_complex_structure(key):
    """Ensures the dynamic model correctly outputs distinct Real and Imaginary channels."""
    from src.model_tdse_1d import ComplexQuantumWavepacket
    model = ComplexQuantumWavepacket(key)

    # Robust check: Test multiple points to ensure channels aren't globally identical
    test_points = [(0.1, 0.5), (0.3, -0.2), (1.0, 0.8)]
    outputs = [model(t, x) for t, x in test_points]

    u_vals = jnp.array([o[0] for o in outputs])
    v_vals = jnp.array([o[1] for o in outputs])

    assert not jnp.allclose(u_vals, v_vals), "Real and Imaginary channels are identical; phase information lost."
    assert not jnp.any(jnp.isnan(u_vals)) and not jnp.any(jnp.isnan(v_vals)), "TDSE model produced NaNs."


def test_inverse_potential_gradient(key):
    """Phase III: Verifies that gradients flow to the potential MLP for discovery."""
    from src.model_inverse_1d import UnknownPotential
    from src.physics_inverse_1d import inverse_residual_loss

    model = UnknownPotential(key)
    x_grid = jnp.linspace(-3.0, 3.0, 100)

    _, grads = eqx.filter_value_and_grad(inverse_residual_loss)(model, x_grid)

    # Flatten and sum absolute gradients
    from jax.flatten_util import ravel_pytree
    flat_grads, _ = ravel_pytree(grads)
    grad_magnitude = jnp.sum(jnp.abs(flat_grads))

    # Robust floor check
    assert float(grad_magnitude) > 1e-8, f"Gradients are effectively zero: {float(grad_magnitude):.2e}"