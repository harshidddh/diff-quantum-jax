import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_tdse_1d import ComplexQuantumWavepacket
from physics_tdse_1d import physics_loss


# The Potential Barrier we want the wave to crash into
def gaussian_barrier(x):
    return 2.5 * jnp.exp(-2.0 * x ** 2)


def main():
    print("Initializing Spacetime Meshgrid...")
    # Space: -6 to 6. Time: 0 to 1.0 seconds
    x_coords = jnp.linspace(-6.0, 6.0, 200)
    t_coords = jnp.linspace(0.0, 2.5, 80)

    # Create the Spacetime Collocation Points (flattened grid)
    T, X = jnp.meshgrid(t_coords, x_coords)
    t_collocation = T.flatten()
    x_collocation = X.flatten()

    # Points specifically for the initial condition at t=0
    x_initial = jnp.linspace(-6.0, 6.0, 400)

    # Initialize Model and Optimizer
    key = jax.random.PRNGKey(42)
    model = ComplexQuantumWavepacket(key)

    optimizer = optax.adam(learning_rate=3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, o_s, t_col, x_col, x_init):
        loss, grads = eqx.filter_value_and_grad(physics_loss)(m, t_col, x_col, x_init, gaussian_barrier)
        updates, o_s = optimizer.update(grads, o_s, m)
        m = eqx.apply_updates(m, updates)
        return m, o_s, loss

    epochs = 6000
    print("\n--- Training Quantum Dynamics (TDSE) ---")
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, t_collocation, x_collocation, x_initial)
        if epoch % 400 == 0:
            print(f"Epoch {epoch:04d} | PDE + IC Loss: {loss:.5f}")

    print(f"Final Loss: {loss:.5f}")

    # --- Plotting the Dynamics (Time Snapshots) ---
    print("\nGenerating Quantum Tunneling Snapshots...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time_snapshots = [0.0, 1.25, 2.5]

    V_x = gaussian_barrier(x_initial)

    for ax, t_val in zip(axes, time_snapshots):
        # Ask the neural network what the wave looks like at time 't'
        vmap_model = jax.vmap(lambda x: model.psi_squared(t_val, x))
        prob_density = vmap_model(x_initial)

        ax.plot(x_initial, V_x, 'k:', alpha=0.6, label="Potential Barrier")
        ax.plot(x_initial, prob_density, 'b-', linewidth=2, label=f"$|\psi|^2$ at $t={t_val}$")

        ax.set_ylim(-0.1, 1.5)
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Position $x$")
    plt.suptitle("PINN: Time-Dependent Quantum Wavepacket Scattering")
    plt.tight_layout()

    plt.savefig("quantum_dynamics_snapshots.png", dpi=300)
    plt.close()
    print("Saved time evolution plot as quantum_dynamics_snapshots.png!")


if __name__ == "__main__":
    main()