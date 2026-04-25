import jax
import jax.numpy as jnp
import equinox as eqx


class QuantumWaveFunction(eqx.Module):
    mlp: eqx.nn.MLP
    is_odd: bool  # Flag to enforce physical symmetry

    def __init__(self, key, is_odd=False):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=32, depth=3,
            activation=jax.nn.gelu, key=key
        )
        self.is_odd = is_odd

    def __call__(self, x):
        # Evaluate the network at x and -x
        val_pos = self.mlp(jnp.array([x]))[0]
        val_neg = self.mlp(jnp.array([-x]))[0]

        # Wider envelope for the double-well
        envelope = jnp.exp(-0.2 * x ** 2)

        # Apply hard topological constraints
        if self.is_odd:
            # Forced Antisymmetric (Odd) Function: f(x) = -f(-x) -> guaranteed 0 at x=0
            return (val_pos - val_neg) * envelope
        else:
            # Forced Symmetric (Even) Function: f(x) = f(-x)
            return (val_pos + val_neg) * envelope