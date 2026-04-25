import jax
import jax.numpy as jnp
import equinox as eqx

class QuantumWaveFunction(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
    # 1 input (position x), 1 output (wavefunction amplitude psi)
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=32,
            depth=3,
            activation=jax.nn.gelu,
            key=key
        )

    def __call__(self, x):
        # the raw neural network output
        nn_out = self.mlp(jnp.array([x]))[0]
        # Multiply by a Gaussian envelope to enforce boundary conditions naturally
        envelope = jnp.exp(-0.5 * x**2)
        return nn_out * envelope
