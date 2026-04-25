import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_inverse_1d import UnknownPotential
from physics_inverse_1d import inverse_residual_loss


def main():
    # Define the domain where we collected our data
    x_grid = jnp.linspace(-3.0, 3.0, 500)

    key = jax.random.PRNGKey(42)
    model = UnknownPotential(key)

    # Slightly higher learning rate for regression
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, o_s, grid):
        loss, grads = eqx.filter_value_and_grad(inverse_residual_loss)(m, grid)
        updates, o_s = optimizer.update(grads, o_s, m)
        m = eqx.apply_updates(m, updates)
        return m, o_s, loss

    print("--- Training Inverse PINN: Discovering V(x) ---")
    epochs = 1500
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, x_grid)
        if epoch % 300 == 0:
            print(f"Epoch {epoch:04d} | Physics Residual Loss: {loss:.6f}")

    print(f"Final Loss: {loss:.6f}")

    # --- Plotting the Scientific Discovery ---
    print("\nVisualizing AI-Discovered Potential...")
    vmap_V = jax.vmap(model)
    V_pred = vmap_V(x_grid)

    # The Theoretical Truth (Harmonic Oscillator)
    V_true = 0.5 * x_grid ** 2

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, V_true, 'k--', linewidth=2, label=r"True Hidden Potential $V(x) = \frac{1}{2}x^2$")
    plt.plot(x_grid, V_pred, 'r-', linewidth=2, alpha=0.8, label=r"AI Discovered Potential $V_\theta(x)$")

    plt.title("Inverse Quantum Mechanics: Data-Driven Discovery of the Harmonic Oscillator")
    plt.xlabel("Position $x$")
    plt.ylabel("Potential Energy $V(x)$")
    plt.legend(loc='lower center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("results/inverse_discovery.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot successfully saved as results/inverse_discovery.png!")


if __name__ == "__main__":
    main()