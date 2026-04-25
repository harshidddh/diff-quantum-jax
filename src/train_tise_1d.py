import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_tise_1d import QuantumWaveFunction
from physics_tise_1d import variational_loss, double_well_potential


def train_state(key_seed, is_odd, state_name, epochs=1500):
    key = jax.random.PRNGKey(key_seed)
    # Initialize with the hard constraint!
    model = QuantumWaveFunction(key, is_odd=is_odd)

    optimizer = optax.adam(learning_rate=2e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, o_s, grid):
        loss, grads = eqx.filter_value_and_grad(variational_loss)(m, grid, double_well_potential)
        updates, o_s = optimizer.update(grads, o_s, m)
        m = eqx.apply_updates(m, updates)
        return m, o_s, loss

    print(f"\n--- Optimizing {state_name} ---")
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, x_grid)
        if epoch % 300 == 0:
            print(f"Epoch {epoch:04d} | Energy: {loss:.5f}")

    print(f"Final {state_name} Energy: {loss:.5f}")
    return model


x_grid = jnp.linspace(-6.0, 6.0, 1500)


def main():
    x_grid = jnp.linspace(-6.0, 6.0, 1500)

    # Check these lines carefully!
    ground_model = train_state(42, is_odd=False, state_name="Ground State")
    excited_model = train_state(99, is_odd=True, state_name="1st Excited State")
    # 1. Train Ground State (Even Symmetry)
    ground_model = train_state(42, is_odd=False, state_name="Ground State")

    # 2. Train Excited State (Odd Symmetry)
    excited_model = train_state(99, is_odd=True, state_name="1st Excited State")

    # 3. Visualization
    print("\nGenerating final validated plot...")
    vmap_ground = jax.vmap(ground_model)
    vmap_excited = jax.vmap(excited_model)
    vmap_pot = jax.vmap(double_well_potential)

    psi_0 = vmap_ground(x_grid)
    psi_1 = vmap_excited(x_grid)
    V_x = vmap_pot(x_grid)

    psi_0 /= jnp.sqrt(jnp.trapezoid(psi_0 ** 2, x=x_grid))
    psi_1 /= jnp.sqrt(jnp.trapezoid(psi_1 ** 2, x=x_grid))

    plt.figure(figsize=(12, 7))
    plt.plot(x_grid, V_x, 'k:', alpha=0.6, linewidth=2, label=r"Double-Well Potential $V(x)$")

    # Shifted for visual clarity
    plt.plot(x_grid, psi_0 ** 2 + 0.3, 'b-', linewidth=2.5, label=r"Ground State $|\psi_0|^2$")
    plt.plot(x_grid, psi_1 ** 2 + 1.2, 'r-', linewidth=2.5, label=r"1st Excited State $|\psi_1|^2$")

    plt.title("PINN: Hard Constrained Double-Well Quantum States")
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Energy Level & Probability Density")
    plt.ylim(-0.2, 3.0)
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("double_well_hard_constraints.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot successfully saved as double_well_hard_constraints.png!")


if __name__ == "__main__":
    main()