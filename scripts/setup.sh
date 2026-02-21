#!/bin/bash
# NOTE: no 'set -e' — script is sourced with '.' which would kill the shell on error

###############################################################################
# EXPO Adaptive Beta — RunPod Setup Script
#
# Verified-compatible version set (all cross-checked):
#   python=3.10, jax==0.4.35, jaxlib==0.4.34, flax==0.7.5, optax==0.1.5,
#   chex==0.1.86, distrax==0.1.5, tfp==0.19.0, gym==0.23.1
#
# Uses system CUDA (cuda12_local) — NOT pip nvidia-* packages.
# Requires: CUDA 12.x + cuDNN 9.x pre-installed on the machine.
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
# Step 0: System CUDA environment (persists because script is sourced)
# ---------------------------------------------------------------------------
# cuda12_local expects system CUDA. Set paths so JAX + XLA can find it.
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export PATH="$CUDA_ROOT/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"

# MuJoCo 210 path (needed by mujoco_py, used by D4RL)
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[0/6] CUDA environment:"
echo "       CUDA_ROOT=$CUDA_ROOT"
echo "       nvcc: $(nvcc --version 2>/dev/null | grep 'release' || echo 'not found')"

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
# Step 2: Install Python 3.10 (D4RL requires python_requires<3.11)
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
# Step 3: Create venv + ensure clean state
# ---------------------------------------------------------------------------
echo "[3/6] Setting up Python 3.10 virtual environment..."

# If venv exists, check critical package versions. Nuke if any are wrong.
if [ -d "$VENV" ] && [ -f "$PIP" ]; then
    INSTALLED_JAX=$("$PIP" show jax 2>/dev/null | grep "^Version:" | awk '{print $2}')
    INSTALLED_FLAX=$("$PIP" show flax 2>/dev/null | grep "^Version:" | awk '{print $2}')
    if [ "$INSTALLED_JAX" != "0.4.35" ] || [ "$INSTALLED_FLAX" != "0.8.5" ]; then
        echo "       Stale venv (JAX=$INSTALLED_JAX, flax=$INSTALLED_FLAX), recreating..."
        rm -rf "$VENV"
    else
        echo "       Existing venv OK (JAX=$INSTALLED_JAX, flax=$INSTALLED_FLAX)"
    fi
fi

if [ ! -d "$VENV" ]; then
    python3.10 -m venv "$VENV"
fi

echo "       Python: $PYTHON"
echo "       $($PYTHON --version)"

"$PIP" install --upgrade pip setuptools wheel 2>&1 | tail -1

# ---------------------------------------------------------------------------
# Step 4: Install JAX 0.4.35 with system CUDA (cuda12_local)
# ---------------------------------------------------------------------------
echo "[4/6] Installing JAX 0.4.35 with system CUDA 12..."

# cuda12_local = system CUDA. cuda12_pip = pip nvidia-* packages.
# pip nvidia packages have __file__=None bug on RunPod/cloud (Python 3.10).
# Note: JAX 0.4.35 has packaging bug #24826 — it pulls jaxlib==0.4.34 (fine).
"$PIP" install "jax[cuda12_local]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Remove any leftover pip nvidia packages from previous cuda12_pip installs.
# These conflict with system CUDA. Do NOT remove jax-cuda12-plugin/pjrt.
echo "       Cleaning up pip nvidia packages (using system CUDA instead)..."
"$PIP" uninstall -y \
    nvidia-cuda-nvcc-cu12 nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 2>/dev/null || true

# Smoke test: can JAX see the GPU?
echo "       JAX smoke test..."
"$PYTHON" -c "import jax; devs=jax.devices(); print(f'       JAX {jax.__version__}, devices: {devs}')" 2>&1 || {
    echo "       ERROR: JAX cannot initialize. Check CUDA installation."
    echo "       Expected: nvcc --version shows CUDA 12.x"
    echo "       Expected: ls $CUDA_ROOT/lib64/libcudnn* finds cuDNN"
    return 1 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Step 5: Install EXPO dependencies (all versions cross-verified)
# ---------------------------------------------------------------------------
echo "[5/6] Installing EXPO dependencies..."

# Core ML stack — EVERY version pinned to prevent pip from upgrading JAX.
# CRITICAL: flax>=0.8.0 needed (0.7.5 uses define_bool_state removed in JAX 0.4.25+)
#           flax<=0.10.4 needed (0.10.5+ requires jax>=0.5.1)
# CRITICAL: distrax==0.1.5 because 0.1.6+ requires jax>=0.7.0
# CRITICAL: jax==0.4.35 re-stated so flax/chex can't upgrade it
"$PIP" install \
    "jax==0.4.35" \
    "flax==0.8.5" \
    "optax==0.1.5" \
    "chex==0.1.86" \
    "distrax==0.1.5" \
    "tensorflow-probability==0.19.0" \
    ml_collections \
    orbax-checkpoint

# RL environment
"$PIP" install \
    "gym==0.23.1" \
    mujoco \
    dm_control

# MuJoCo 210 + mujoco_py (needed by D4RL's antmaze and locomotion envs)
MUJOCO_DIR="$HOME/.mujoco"
if [ ! -d "$MUJOCO_DIR/mujoco210" ]; then
    echo "       Downloading MuJoCo 210..."
    mkdir -p "$MUJOCO_DIR"
    wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
    tar -xzf /tmp/mujoco210.tar.gz -C "$MUJOCO_DIR/"
    rm /tmp/mujoco210.tar.gz
else
    echo "       MuJoCo 210 already at $MUJOCO_DIR/mujoco210"
fi
export LD_LIBRARY_PATH="$MUJOCO_DIR/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# System deps for mujoco_py compilation
echo "       Installing mujoco_py build deps..."
apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx patchelf 2>/dev/null || true

"$PIP" install "Cython<3" "mujoco_py==2.1.2.14"

# dmcgym: --no-deps to avoid gym[mujoco] pulling legacy mujoco_py again
"$PIP" install --no-deps dmcgym@git+https://github.com/ikostrikov/dmcgym

# D4RL: --no-deps to avoid pulling pybullet (mujoco_py is now installed)
"$PIP" install --no-deps \
    d4rl@git+https://github.com/Farama-Foundation/D4RL.git@2b96431a0e9fd90c8032624b0dc3cd4514d15632

# D4RL's actual runtime deps (that --no-deps skipped)
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
# Step 6: Verify everything
# ---------------------------------------------------------------------------
echo "[6/6] Verifying installation..."

"$PYTHON" -c "
import sys
print(f'Python:         {sys.version.split()[0]}')

import jax, jaxlib
print(f'JAX:            {jax.__version__}')
print(f'jaxlib:         {jaxlib.__version__}')
print(f'JAX devices:    {jax.devices()}')
print(f'GPU available:  {len(jax.devices(\"gpu\")) > 0}')

import flax
print(f'Flax:           {flax.__version__}')

import optax
print(f'Optax:          {optax.__version__}')

import chex
print(f'Chex:           {chex.__version__}')

import distrax
print(f'Distrax:        {distrax.__version__}')

import tensorflow_probability.substrates.jax as tfp
print(f'TFP (JAX):      {tfp.__version__}')

import gym
print(f'Gym:            {gym.__version__}')

try:
    import d4rl
    print('D4RL:           OK')
except Exception as e:
    print(f'D4RL:           WARNING - {e}')

from expo.agents.sac.edit_distance import EditDistance
print('EditDistance:    OK')
print()
print('All checks passed.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Run experiments: bash scripts/run_all.sh"
echo "  Monitor:        watch nvidia-smi"
echo "  Wandb dashboard: https://wandb.ai"
echo "============================================"
