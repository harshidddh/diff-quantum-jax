import jax
import jax.numpy as jnp
import equinox as eqx


class QuantumWaveFunction(eqx.Module):
    mlp: eqx.nn.MLP
    is_odd: bool

    def __init__(self, key, is_odd=False):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=32, depth=3,
            activation=jax.nn.gelu, key=key
        )
        self.is_odd = is_odd
        # Debug print to terminal
        print(f"DEBUG: Initialized model with is_odd={self.is_odd}")

    def __call__(self, x):
        # Force x to be a float for jax.grad
        x = jnp.array(x).reshape()

        val_pos = self.mlp(jnp.array([x]))[0]
        val_neg = self.mlp(jnp.array([-x]))[0]

        envelope = jnp.exp(-0.2 * x ** 2)

        if self.is_odd:
            # This MUST produce 0 at x=0: (f(0) - f(-0)) = 0
            return (val_pos - val_neg) * envelope
        else:
            return (val_pos + val_neg) * envelope