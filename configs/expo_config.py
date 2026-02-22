from configs import sac_config


def get_config():
    config = sac_config.get_config()


    config.model_cls = "EXPOLearner"


    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm=True

    config.N = 32
    config.n_edit_samples = 0
    config.entropy_scale = 1.0
    config.edit_action_scale = 1.0
    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    config.batch_split = 1
    config.T = 10

    config.adaptive_beta = False
    config.beta_lr = 1e-4
    config.beta_warmup_steps = 0

    return config
