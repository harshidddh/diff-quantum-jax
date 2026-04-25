import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from model import QuantumWavefunction
from physics import variational_loss


def analytical_ground_state(x):
    # The exact theoretical quantum mechanical solution
    return (1.0 / jnp.pi) ** 0.25 * jnp.exp(-0.5 * x ** 2)


def main():
    # 1. Initialize Model and Optimizer
    key = jax.random.PRNGKey(42)
    model = QuantumWavefunction(key)

    # Optax Adam optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 2. Define the computational grid (integration domain)
    x_grid = jnp.linspace(-5.0, 5.0, 1000)

    # 3. JIT-compiled training step
    @eqx.filter_jit
    def step(model, opt_state, grid):
        # Calculate loss and exact gradients simultaneously
        loss, grads = eqx.filter_value_and_grad(variational_loss)(model, grid)

        # Apply gradients
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    # 4. Training Loop
    epochs = 1000  # 1000 iterations for clean convergence
    print("Initiating PINN Optimization (Target Energy: E = 0.500)")

    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, x_grid)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Variational Energy: {loss:.5f}")

    print(f"Final Variational Energy: {loss:.5f}")

    # 5. Validation Plotting
    print("Generating analytical validation plot...")
    vmap_model = jax.vmap(model)
    psi_pinn = vmap_model(x_grid)

    # Normalize the PINN wavefunction
    psi_pinn_normalized = psi_pinn / jnp.sqrt(jnp.trapz(psi_pinn ** 2, x=x_grid))
    psi_exact = analytical_ground_state(x_grid)

    # Plot Probability Densities |psi|^2
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, psi_exact ** 2, 'k--', linewidth=2, label="Analytical Truth $|\psi_0|^2$")
    plt.plot(x_grid, psi_pinn_normalized ** 2, 'r-', linewidth=2, alpha=0.8,
             label="PINN Approximation $|\psi_\\theta|^2$")

    # Plot the potential V(x) for context (scaled down for visual clarity)
    V_x = 0.5 * x_grid ** 2
    plt.plot(x_grid, V_x * 0.1, 'gray', linestyle=':', label="Potential $V(x)$ (scaled)")

    plt.title("Quantum Harmonic Oscillator: Ground State Discovery via PINN")
    plt.xlabel("Position $x$")
    plt.ylabel("Probability Density $|\psi(x)|^2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and show
    plt.savefig("qho_result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()