# Expressive Policy Optimization (EXPO)

![alt text](plots/offline_to_online.png "Title")

Code for the paper "EXPO: Stable Reinforcement Learning with Expressive Policies", available [here](https://arxiv.org/abs/2507.07986)

This code is built on top of the [jaxrl](https://github.com/ikostrikov/jaxrl) framework and the [RLPD](https://github.com/ikostrikov/rlpd
) repository. 

# Installation

```bash
conda env create -f environment.yml
conda activate expo
conda install patchelf  # If you use conda.
pip install -r requirements.txt
conda deactivate
conda activate expo
```

# RunPod / VM Setup

For RunPod or other cloud VMs, use the one-line setup script:

```bash
git clone -b adaptive-beta https://github.com/andamanopal/EXPO.git /workspace/EXPO
cd /workspace/EXPO && . scripts/setup.sh
```

JAX uses pip-bundled CUDA packages (`cuda12_pip`) which are self-contained. **Do not** set `XLA_FLAGS="--xla_gpu_cuda_data_dir=..."` — this forces JAX to use system CUDA instead of its own pip packages, causing version conflicts with cuDNN.

# Experiments

Example scripts are provided in the scripts directory. All commands below use the venv from `setup.sh` and include the required environment variables — copy and paste the entire block.

## D4RL Antmaze

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_finetuning.py --env_name=antmaze-large-play-v2 \
    --seed=3 --utd_ratio=20 --start_training 5000 --max_steps 300000 \
    --expo=True --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 --project_name=expo
```

## Adroit Binary

First, download and unzip `.npy` files into `~/.datasets/awac-data/` from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view).

Make sure you have `mjrl` installed:
```bash
git clone https://github.com/aravindr93/mjrl
cd mjrl
pip install -e .
```

Then, recursively clone `mj_envs` from this fork:
```bash
git clone --recursive https://github.com/philipjball/mj_envs.git
```

Then sync the submodules (add the `--init` flag if you didn't recursively clone):
```bash
$ cd mj_envs
$ git submodule update --remote
```

Finally:
```bash
$ pip install -e .
```
May need to remove mujoco dependency in setup.py of mj_env

## Adaptive Beta

EXPO supports Q-advantage adaptive beta, which automatically tunes the edit action scale (`beta`) by maximizing Q-values. Instead of hand-tuning `edit_action_scale` per environment, enable `adaptive_beta` and let the critic guide beta to the optimal value.

### Run all experiments

The easiest way is `scripts/run_all.sh`, which runs 4 experiments sequentially: two fixed-beta baselines and two adaptive-beta runs.

```bash
# Quick validation (~10K steps each, confirms code runs without errors)
bash scripts/run_all.sh --quick

# Full runs (antmaze 300K steps, pen 1M steps)
bash scripts/run_all.sh
```

### Individual experiments

**Baseline — fixed beta (antmaze)**
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_finetuning.py --env_name=antmaze-large-diverse-v2 \
    --seed=42 --utd_ratio=20 --start_training 5000 --max_steps 300000 \
    --expo=True --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 --project_name=expo-adaptive-beta
```

**Baseline — fixed beta (pen)**
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_finetuning.py --env_name=pen-binary-v0 \
    --seed=42 --utd_ratio=20 --start_training 0 --max_steps 1000000 \
    --expo=True --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.7 --config.actor_drop=0.1 \
    --project_name=expo-adaptive-beta
```

**Adaptive beta — antmaze (should converge toward ~0.05)**
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_finetuning.py --env_name=antmaze-large-diverse-v2 \
    --seed=42 --utd_ratio=20 --start_training 5000 --max_steps 300000 \
    --expo=True --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.3 --config.adaptive_beta=True \
    --project_name=expo-adaptive-beta
```

**Adaptive beta — pen (should converge toward ~0.70)**
```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_finetuning.py --env_name=pen-binary-v0 \
    --seed=42 --utd_ratio=20 --start_training 0 --max_steps 1000000 \
    --expo=True --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.3 --config.adaptive_beta=True \
    --config.actor_drop=0.1 --project_name=expo-adaptive-beta
```

### Plotting results

```bash
PYTHON=".venv/bin/python" && \
$PYTHON scripts/plot_results.py --wandb_project expo-adaptive-beta --output_dir plots/results
```

### Key wandb metrics to monitor

| Metric | What to expect |
|--------|---------------|
| `training/beta` | Changes over time (not stuck at init) |
| `training/q_advantage` | Positive when beta is growing, negative when shrinking |
| `evaluation/beta` | Converges toward ~0.05 (antmaze) or ~0.70 (pen) |
| `evaluation/return` | Matches or beats fixed-beta baselines |

## Robomimic

train_robo.py is used fro Robomimic and MimicGen environments.

Download the datasets from [here](https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html) and put it in ./robomimic/datasets/{env_name}/ph for ph and robomimic/datasets/{env_name}/mh for mh. 


```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_robo.py --env_name=square \
    --seed=3 --utd_ratio=20 --dataset_dir='ph' \
    --start_training 5000 --max_steps 500000 \
    --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 --project_name=expo
```


## MimicGen

Install MimicGen by running the following command

```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```


Download the datasets from [here](https://drive.google.com/file/d/1qemTmLkEkE17dFN6A2BJvVYLJBtLbN67/view?usp=sharing) and put it in ./mimicgen/datasets/{env_name}/. The datasets are subsampled from the original MimicGen datasets. 


Then, run with 

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export D4RL_SUPPRESS_IMPORT_ERROR=1 && \
PYTHON=".venv/bin/python" && \
$PYTHON train_robo.py --env_name=threading \
    --seed=3 --utd_ratio=20 --start_training 5000 --max_steps 500000 \
    --config=configs/expo_config.py \
    --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
    --config.N=8 --config.n_edit_samples=8 \
    --config.edit_action_scale=0.05 --project_name=expo
```


## Citation


```bash
@misc{dong2025expo,
      title={EXPO: Stable Reinforcement Learning with Expressive Policies}, 
      author={Perry Dong and Qiyang Li and Dorsa Sadigh and Chelsea Finn},
      year={2025},
      eprint={2507.07986},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.07986}, 
}
```

