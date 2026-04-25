import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_tise_2d import QuantumCorral2D
from physics_tise_2d import variational_loss_2d


def main():
    print("Initializing 2D Spatial Meshgrid...")
    L = 2.0  # The box goes from -2 to 2
    N = 60  # 60x60 grid (3600 spatial points)

    x_coords = jnp.linspace(-L, L, N)
    y_coords = jnp.linspace(-L, L, N)

    key = jax.random.PRNGKey(42)
    # Initialize the 2D model with the boundary L=2.0
    model = QuantumCorral2D(key, L=L)

    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, o_s, x_g, y_g):
        loss, grads = eqx.filter_value_and_grad(variational_loss_2d)(m, x_g, y_g)
        updates, o_s = optimizer.update(grads, o_s, m)
        m = eqx.apply_updates(m, updates)
        return m, o_s, loss

    epochs = 1500
    print("\n--- Training 2D Quantum Corral ---")
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, x_coords, y_coords)
        if epoch % 300 == 0:
            print(f"Epoch {epoch:04d} | 2D Variational Energy: {loss:.5f}")

    print(f"Final 2D Energy: {loss:.5f}")

    # --- Plotting the 2D Orbital Heatmap ---
    print("\nGenerating 2D Quantum Orbital Heatmap...")
    X, Y = jnp.meshgrid(x_coords, y_coords)

    # Flatten the grid to pass through the neural network
    flat_x = X.flatten()
    flat_y = Y.flatten()

    # Evaluate the wavefunction
    vmap_model = jax.vmap(model)
    psi_flat = vmap_model(flat_x, flat_y)

    # Reshape back to 2D and calculate Probability Density
    psi_2d = psi_flat.reshape(N, N)
    prob_density = psi_2d ** 2

    # Normalize the 2D density for plotting
    prob_density /= jnp.sum(prob_density)

    plt.figure(figsize=(8, 6))
    # Create a filled contour plot (heatmap)
    cp = plt.contourf(X, Y, prob_density, levels=50, cmap='magma')
    plt.colorbar(cp, label=r'Probability Density $|\psi(x,y)|^2$')

    plt.title("2D Quantum Corral: Ground State Orbital")
    plt.xlabel("Position $x$")
    plt.ylabel("Position $y$")
    plt.tight_layout()

    plt.savefig("results/quantum_corral_2d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot successfully saved as results/quantum_corral_2d.png!")


if __name__ == "__main__":
    main()