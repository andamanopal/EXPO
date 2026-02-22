"""Learnable beta (edit action scale) via Q-advantage gradient ascent.

Mirrors the Temperature module pattern used for SAC alpha tuning.
Beta is optimized by maximizing Q(s, a_base + beta * delta): when
increasing beta improves Q-values, beta grows; when edits overshoot,
beta shrinks. Equilibrium is environment-specific by construction.

Uses sigmoid reparameterization instead of exp+clip to guarantee
nonzero gradients at all beta values within [min_beta, max_beta].
"""

import jax
import flax.linen as nn
import jax.numpy as jnp


class EditDistance(nn.Module):
    initial_beta: float = 0.1
    min_beta: float = 0.01
    max_beta: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        init_sigmoid = (self.initial_beta - self.min_beta) / (self.max_beta - self.min_beta)
        init_logit = jnp.log(init_sigmoid / (1.0 - init_sigmoid))

        logit = self.param(
            "logit",
            init_fn=lambda key: jnp.full((), init_logit),
        )
        return jax.nn.sigmoid(logit) * (self.max_beta - self.min_beta) + self.min_beta
