#!/bin/bash
# NOTE: no 'set -e' — script is sourced with '.' which would kill the shell on error

###############################################################################
# EXPO Adaptive Beta — RunPod Setup Script
#
# Pinned versions from EXPO environment.yml:
#   jax=0.4.*, optax==0.1.5, tensorflow-probability==0.19.0, gym==0.23.1
#
# Usage:
#   git clone -b adaptive-beta https://github.com/andamanopal/EXPO.git /data/EXPO
#   cd /data/EXPO && . scripts/setup.sh
###############################################################################

echo "============================================"
echo "  EXPO Adaptive Beta — Environment Setup"
echo "============================================"

WORKDIR="${EXPO_WORKDIR:-/data/EXPO}"
VENV="$WORKDIR/.venv"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# ---------------------------------------------------------------------------
# Step 1: Check repo
# ---------------------------------------------------------------------------
if [ ! -f "$WORKDIR/train_finetuning.py" ]; then
    echo "[1/6] ERROR: EXPO repo not found at $WORKDIR"
    echo "       git clone -b adaptive-beta https://github.com/andamanopal/EXPO.git $WORKDIR"
    return 1 2>/dev/null || true
else
    echo "[1/6] EXPO found at $WORKDIR"
fi

cd "$WORKDIR"

# ---------------------------------------------------------------------------
# Step 2: Install Python 3.10 (D4RL requires <3.11)
# ---------------------------------------------------------------------------
echo "[2/6] Ensuring Python 3.10..."

if ! command -v python3.10 &> /dev/null; then
    echo "       Installing python3.10 via apt..."
    apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv python3.10-dev 2>/dev/null
fi

if ! command -v python3.10 &> /dev/null; then
    echo "       ERROR: python3.10 not available. Install it manually."
    return 1 2>/dev/null || true
fi

echo "       $(python3.10 --version)"

# ---------------------------------------------------------------------------
# Step 3: Create venv with Python 3.10
# ---------------------------------------------------------------------------
echo "[3/6] Creating Python 3.10 virtual environment..."

if [ ! -d "$VENV" ]; then
    python3.10 -m venv "$VENV"
fi

echo "       Python: $PYTHON"
echo "       $($PYTHON --version)"

"$PIP" install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Step 4: Install JAX 0.4.x with CUDA (pinned to match EXPO environment.yml)
# ---------------------------------------------------------------------------
echo "[4/6] Installing JAX 0.4.x with CUDA 12..."

"$PIP" install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------------------------------------------------------
# Step 5: Install EXPO dependencies (versions from environment.yml)
# ---------------------------------------------------------------------------
echo "[5/6] Installing EXPO dependencies..."

# Core ML — pin tfp to 0.19.0 (matches environment.yml, has substrates.jax)
"$PIP" install \
    flax \
    "optax==0.1.5" \
    chex \
    "tensorflow-probability==0.19.0" \
    distrax \
    ml_collections \
    orbax-checkpoint==0.2.3

# RL environment
"$PIP" install \
    "gym==0.23.1" \
    mujoco \
    dm_control

# dmcgym: --no-deps to avoid gym[mujoco] pulling legacy mujoco_py
"$PIP" install --no-deps dmcgym@git+https://github.com/ikostrikov/dmcgym

# D4RL: --no-deps to avoid pulling mujoco_py and pybullet
"$PIP" install --no-deps \
    d4rl@git+https://github.com/Farama-Foundation/D4RL.git@2b96431a0e9fd90c8032624b0dc3cd4514d15632

# D4RL's actual runtime deps (without mujoco_py)
"$PIP" install h5py click termcolor

# Logging and utils
"$PIP" install \
    wandb \
    absl-py \
    tqdm \
    matplotlib \
    seaborn \
    easydict \
    dm_env_wrappers \
    "jax-jumpy==1.0.0" \
    submitit \
    moviepy \
    imageio \
    tensorboardX

# ---------------------------------------------------------------------------
# Step 6: Verify
# ---------------------------------------------------------------------------
echo "[6/6] Verifying installation..."

"$PYTHON" -c "
import sys
print(f'Python:         {sys.version}')

import jax
print(f'JAX version:    {jax.__version__}')
print(f'JAX devices:    {jax.devices()}')
print(f'GPU available:  {len(jax.devices(\"gpu\")) > 0}')

import flax
print(f'Flax version:   {flax.__version__}')

import optax
print(f'Optax version:  {optax.__version__}')

import tensorflow_probability.substrates.jax as tfp
print(f'TFP (JAX):      {tfp.__version__}')

import gym
print(f'Gym version:    {gym.__version__}')

try:
    import d4rl
    print('D4RL:           OK')
except Exception as e:
    print(f'D4RL:           WARNING - {e}')

from expo.agents.sac.edit_distance import EditDistance
print('EditDistance:    OK')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Run experiments: bash scripts/run_all.sh"
echo "  Monitor:        watch nvidia-smi"
echo "  Wandb dashboard: https://wandb.ai"
echo "============================================"
