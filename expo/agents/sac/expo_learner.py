"""Starter code from the RLPD repository https://github.com/ikostrikov/rlpd"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState

import numpy as np

from expo.agents.agent import Agent
from expo.agents.sac.edit_distance import EditDistance
from expo.agents.sac.temperature import Temperature
from expo.data.dataset import DatasetDict
from expo.distributions import TanhNormal
from expo.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)
from expo.networks import DiffusionMLP, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, ddpm_train_sampler, DiffusionMLPResNet, get_weight_decay_mask, vp_beta_schedule


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):

    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


class EXPOLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    target_actor: TrainState
    edit_actor: TrainState
    temp: TrainState
    beta_state: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    clip_sampler: bool = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_action_scale: float = struct.field(pytree_node=False)
    adaptive_beta: bool = struct.field(pytree_node=False)
    beta_warmup_steps: int = struct.field(pytree_node=False)
    update_step: int
    batch_split: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddpm_temperature: float
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: bool
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0, 
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        adjust_target_entropy: bool = False, 
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None, 
        d_actor_drop: Optional[float] = None, 
        T: int = 10, 
        N: int = 32,
        batch_split: int = 1, 
        M: int = 0,
        n_edit_samples: int = 0, 
        edit_action_scale: float = 1.0, 
        actor_layer_norm: bool = True,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_num_blocks: int = 3,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = 'vp',
        adaptive_beta: bool = False,
        beta_lr: float = 1e-4,
        beta_warmup_steps: int = 0,
    ):

        action_dim = action_space.shape[-1]

        if isinstance(action_space, gym.Space):
            observations = observation_space.sample()
            actions = action_space.sample()

        else:
            observations = observation_space
            actions = action_space

        if target_entropy is None:
            target_entropy = -action_dim / 2
            
            if adjust_target_entropy:
                target_entropy = -action_dim / 2 + action_dim * jnp.log(edit_action_scale)


        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, beta_key = jax.random.split(rng, 5)

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(DiffusionMLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(DiffusionMLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=d_actor_drop,
                                    out_dim=action_dim,
                                    activations=nn.swish)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        actor = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr))
        
        target_actor = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        edit_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        edit_actor_def = TanhNormal(edit_actor_base_cls, action_dim)
        edit_observations = jnp.concatenate([observations, jnp.ones((1, action_dim))], axis=1)
        edit_actor_params = edit_actor_def.init(actor_key, edit_observations)["params"]
        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply, 
            params=edit_actor_params, 
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        beta_def = EditDistance(
            initial_beta=edit_action_scale,
            min_beta=0.01,
            max_beta=1.0,
        )
        beta_params = beta_def.init(beta_key)["params"]
        beta_tx = (
            optax.adam(learning_rate=beta_lr)
            if adaptive_beta
            else optax.GradientTransformation(lambda _: None, lambda _: None)
        )
        beta_state = TrainState.create(
            apply_fn=beta_def.apply,
            params=beta_params,
            tx=beta_tx,
        )

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            edit_actor=edit_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            action_dim=action_dim,
            clip_sampler=clip_sampler,
            T=T,
            N=N,
            n_edit_samples=n_edit_samples,
            edit_action_scale=edit_action_scale,
            adaptive_beta=adaptive_beta,
            beta_warmup_steps=beta_warmup_steps,
            update_step=jnp.int32(0),
            batch_split=batch_split,
            M=M,
            actor_tau=actor_tau,
            ddpm_temperature=ddpm_temperature,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            beta_state=beta_state,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )
    

    def get_beta(self):
        if self.adaptive_beta:
            return self.beta_state.apply_fn(
                {"params": self.beta_state.params}
            )
        return self.edit_action_scale

    def eval_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        actor_params = self.target_actor.params
        actions, rng = ddpm_sampler(self.actor.apply_fn, actor_params, self.T, rng, self.action_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)

        diffusion_actions = actions

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate([observations, jnp.expand_dims(observations[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)

                r_observations = jnp.repeat(jnp.expand_dims(observations[0], axis = 0), self.n_edit_samples, axis=0)
                d_actions = diffusion_actions.copy()[:self.n_edit_samples]
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
                r_samples = r_samples * self.get_beta() + d_actions
                actions = jnp.concatenate([actions, r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)
            idx = jnp.argmax(qs)
            action = actions[idx]

        else:

            action = actions[0]

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze()), self.replace(rng=rng)


    def sample_batch_actions(self, observations):
        rng = self.rng

        observations = jnp.squeeze(observations)
        observations = jax.device_put(observations)
        
        batch_size = observations.shape[0]
        observations_repeated = jnp.repeat(observations, self.N, axis=0)

        actor_params = self.actor.params
        actions_flat, rng = ddpm_train_sampler(
            self.actor.apply_fn, actor_params, self.T, rng, self.action_dim, 
            observations_repeated, self.alphas, self.alpha_hats, self.betas, 
            self.ddpm_temperature, self.M, self.clip_sampler
        )

        actions = actions_flat.reshape(batch_size, self.N, -1)
        observations_repeated = jnp.repeat(observations, self.N + self.n_edit_samples, axis=0)
        
        if self.n_edit_samples > 0:
            key, rng = jax.random.split(rng, 2)
            r_observations = jnp.repeat(observations, self.n_edit_samples, axis=0) # (batch_size * n_edit_samples, obs_dim) repeats
            d_actions = actions.copy()[:, :self.n_edit_samples].reshape(-1, actions.shape[-1])
            r_observations = jnp.concatenate([r_observations, d_actions], axis=1) # self.n_edit_samples actions for each observation
            r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
            r_samples = r_samples * self.get_beta() + d_actions
            actions = jnp.concatenate([actions, r_samples.reshape(batch_size, self.n_edit_samples, -1)], axis=1)
            actions_flat = actions.reshape(-1, actions.shape[-1])


        
        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )
            
            obs_flat = observations_repeated
            actions_flat = actions_flat

            qs = compute_q(self.target_critic.apply_fn, target_params, obs_flat, actions_flat)
            qs = qs.reshape(batch_size, self.N + self.n_edit_samples)

            best_indices = jnp.argmax(qs, axis=1) 
            batch_indices = jnp.arange(batch_size)
            best_actions = actions[batch_indices, best_indices] 
        else:
            best_actions = actions[:, 0] 
        
        rng, _ = jax.random.split(rng, 2)
        return jnp.array(best_actions.squeeze())
    
    

    def sample_actions(self, observations):
        rng = self.rng
        observations = jnp.squeeze(observations)
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        actor_params = self.target_actor.params
        actions, rng = ddpm_sampler(self.actor.apply_fn, actor_params, self.T, rng, self.action_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)

        diffusion_actions = actions


        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:

                key, rng = jax.random.split(rng, 2)

                observations = jnp.concatenate([observations, jnp.expand_dims(observations[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)

                r_observations = jnp.repeat(jnp.expand_dims(observations[0], axis = 0), self.n_edit_samples, axis=0)
                d_actions = diffusion_actions.copy()[:self.n_edit_samples]
                r_observations = jnp.concatenate([r_observations, d_actions], axis=1)
                r_samples, rng =  _sample_actions(key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations)
                r_samples = r_samples * self.get_beta() + d_actions
                actions = jnp.concatenate([actions, r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, observations, actions)
            idx = jnp.argmax(qs)
            action = actions[idx]

        else:
        
            action = actions[0]

        rng, _ = jax.random.split(rng, 2)
        return np.array(action.squeeze()), self.replace(rng=rng)
    

    def update_edit_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            edit_observations = jnp.concatenate([batch["observations"], batch["actions"]], axis=1)
            dist = self.edit_actor.apply_fn({"params": actor_params}, edit_observations, training=True, rngs={"dropout": dropout_key},)
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)
            beta = self.get_beta()
            actions = actions * beta
            edit_mag = jnp.abs(actions).mean()  # post-beta: depends on beta
            log_probs -= actions.shape[-1] * jnp.log(beta)

            actions = actions + batch["actions"]

            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            edit_actor_loss = (
                self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return edit_actor_loss, {
                "edit_q": q.mean(),
                "edit_actor_loss": edit_actor_loss,
                "entropy": -log_probs.mean(),
                "edit_mag": edit_mag,
                "beta": beta,
            }

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
    

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, self.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], self.action_dim))
        
        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = self.actor.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        agent = self.replace(actor=actor)
        target_score_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.actor_tau
        )

        target_score_model = self.target_actor.replace(params=target_score_params)
        new_agent = self.replace(actor=actor, target_actor=target_score_model, rng=rng)

        return new_agent, info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temp_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_beta(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        dropout_key, rng = jax.random.split(rng)

        # Forward pass: sample raw edits from current edit actor
        edit_obs = jnp.concatenate([batch["observations"], batch["actions"]], axis=1)
        dist = self.edit_actor.apply_fn({"params": self.edit_actor.params}, edit_obs)
        raw_edits = dist.sample(seed=key)

        # Q(s, a_base) baseline — same dropout key as q_edited for low-variance advantage
        q_base_all = self.critic.apply_fn(
            {"params": self.critic.params},
            batch["observations"], batch["actions"], True, rngs={"dropout": dropout_key},
        )
        q_base = jax.lax.stop_gradient(q_base_all.min(axis=0).mean())

        # Stop-gradient everything except beta_params
        raw_edits_sg = jax.lax.stop_gradient(raw_edits)
        obs_sg = jax.lax.stop_gradient(batch["observations"])
        base_sg = jax.lax.stop_gradient(batch["actions"])
        critic_params_sg = jax.lax.stop_gradient(self.critic.params)

        def beta_loss_fn(beta_params):
            beta = self.beta_state.apply_fn({"params": beta_params})
            edited = raw_edits_sg * beta + base_sg

            qs = self.critic.apply_fn(
                {"params": critic_params_sg}, obs_sg, edited,
                True, rngs={"dropout": dropout_key},
            )
            q_edited = qs.min(axis=0).mean()

            loss = -q_edited
            advantage = q_edited - q_base
            return loss, {
                "beta": beta, "beta_loss": loss,
                "q_advantage": advantage, "q_edited": q_edited, "q_base": q_base,
            }

        grads, beta_info = jax.grad(beta_loss_fn, has_aux=True)(self.beta_state.params)

        # Zero grads during warmup
        should_update = self.update_step >= self.beta_warmup_steps
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(should_update, g, jnp.zeros_like(g)), grads
        )

        new_beta_state = self.beta_state.apply_gradients(grads=grads)
        return self.replace(beta_state=new_beta_state, rng=rng), beta_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        next_actions = self.sample_batch_actions(batch["next_observations"])
        rng = self.rng

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info
    

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_edit"))
    def update_offline(self, batch: DatasetDict, utd_ratio: int, pretrain_q: bool, pretrain_edit: bool):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            critic_info = {}
            if pretrain_q:
                new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, diffusion_info = new_agent.update_actor(mini_batch)

        edit_info = {}
        if pretrain_edit:

            if self.n_edit_samples > 0:
                new_agent, edit_info = new_agent.update_edit_actor(mini_batch)
                new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
                edit_info = {**edit_info, **temp_info}

                if self.adaptive_beta:
                    new_agent, beta_info = new_agent.update_beta(mini_batch)
                    edit_info = {**edit_info, **beta_info}

        new_agent = new_agent.replace(update_step=new_agent.update_step + 1)
        return new_agent, {**diffusion_info, **edit_info, **critic_info}
    


    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, diffusion_info = new_agent.update_actor(mini_batch)

        edit_info = {}
        if self.n_edit_samples > 0:
            new_agent, edit_info = new_agent.update_edit_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            edit_info = {**edit_info, **temp_info}

            if self.adaptive_beta:
                new_agent, beta_info = new_agent.update_beta(mini_batch)
                edit_info = {**edit_info, **beta_info}

        new_agent = new_agent.replace(update_step=new_agent.update_step + 1)
        return new_agent, {**diffusion_info, **edit_info, **critic_info}
