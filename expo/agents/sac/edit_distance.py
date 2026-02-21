"""Learnable beta (edit action scale) via dual gradient descent.

Mirrors the Temperature module pattern used for SAC alpha tuning.
When the edit policy's raw output magnitude exceeds the target,
beta decreases to constrain edits. When underutilized, beta increases.
"""

import flax.linen as nn
import jax.numpy as jnp


class EditDistance(nn.Module):
    initial_beta: float = 0.1
    min_beta: float = 0.01
    max_beta: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_beta = self.param(
            "log_beta",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_beta)),
        )
        return jnp.clip(jnp.exp(log_beta), self.min_beta, self.max_beta)
