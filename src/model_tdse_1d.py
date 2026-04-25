import jax
import jax.numpy as jnp
import equinox as eqx


class ComplexQuantumWavepacket(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        # 2 inputs: (time t, position x)
        # 2 outputs: (Real u, Imaginary v)
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=2, width_size=64, depth=4,
            activation=jax.nn.gelu, key=key
        )

    def __call__(self, t, x):
        # The raw neural network outputs
        out = self.mlp(jnp.array([t, x]))
        u, v = out[0], out[1]

        # We enforce boundary conditions: the wavepacket must go to 0 at spatial infinities
        # We do NOT constrain time 't'
        spatial_envelope = jnp.exp(-0.1 * x ** 2)

        return u * spatial_envelope, v * spatial_envelope

    def psi_squared(self, t, x):
        # Helper function to get the actual observable probability density |psi|^2 = u^2 + v^2
        u, v = self(t, x)
        return u ** 2 + v ** 2