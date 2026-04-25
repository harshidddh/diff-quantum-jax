import jax
import jax.numpy as jnp
import equinox as eqx


class QuantumCorral2D(eqx.Module):
    mlp: eqx.nn.MLP
    L: float  # Half-width of the 2D box

    def __init__(self, key, L=2.0):
        # 2 inputs (x, y), 1 output (psi)
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=64, depth=4,
            activation=jax.nn.gelu, key=key
        )
        self.L = L

    def __call__(self, x, y):
        nn_out = self.mlp(jnp.array([x, y]))[0]

        # Hard boundary conditions for a 2D infinite well [-L, L]
        # The wavefunction MUST be zero at the walls
        envelope = (self.L ** 2 - x ** 2) * (self.L ** 2 - y ** 2)

        return nn_out * envelope