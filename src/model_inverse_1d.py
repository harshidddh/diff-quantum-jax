import jax
import jax.numpy as jnp
import equinox as eqx

class UnknownPotential(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        # 1 input (position x), 1 output (Potential Energy V)
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=32, depth=3,
            activation=jax.nn.gelu, key=key
        )

    def __call__(self, x):
        return self.mlp(jnp.array([x]))[0]