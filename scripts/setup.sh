#!/bin/bash
# NOTE: no 'set -e' — script is sourced with '.' which would kill the shell on error

###############################################################################
# EXPO Adaptive Beta — RunPod Setup Script
#
# Verified-compatible version set (all cross-checked):
#   python=3.10, numpy>=1.24<2.0, jax==0.4.35, jaxlib==0.4.34, flax==0.8.5,
#   optax==0.1.5, chex==0.1.86, distrax==0.1.5, tfp==0.19.0, gym==0.23.1
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

# Ensure ALL pip nvidia CUDA packages are installed (cuDNN, ptxas, cublas, etc.)
# Original EXPO uses conda which bundles these automatically. With pip, we need them
# as separate packages since this cloud image lacks a full system CUDA toolkit.
echo "       Ensuring pip nvidia CUDA packages are installed..."
"$PIP" install \
    nvidia-cuda-nvcc-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nccl-cu12 2>/dev/null || true

# Add pip nvidia lib dirs to LD_LIBRARY_PATH (covers both system and pip cuDNN)
NVIDIA_LIBS=$("$PYTHON" -c "
import os, glob
venv = '$VENV'
libs = glob.glob(os.path.join(venv, 'lib/python*/site-packages/nvidia/*/lib'))
print(':'.join(libs)) if libs else print('')
" 2>/dev/null)
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "       Added pip nvidia libs to LD_LIBRARY_PATH"
fi

# Smoke test: can JAX see the GPU?
echo "       JAX smoke test..."
"$PYTHON" -c "import jax; devs=jax.devices(); print(f'       JAX {jax.__version__}, devices: {devs}')" 2>&1 || {
    echo "       ERROR: JAX cannot initialize. Check CUDA installation."
    return 1 2>/dev/null || true
}

# Smoke test 2: can JAX actually compute on GPU? (catches missing cuDNN)
echo "       JAX cuDNN smoke test..."
"$PYTHON" -c "
import jax, jax.numpy as jnp
x = jnp.ones((2, 2))
y = jnp.dot(x, x)
print(f'       cuDNN test: {y.shape} on {y.devices()} — OK')
" 2>&1 || {
    echo "       ERROR: JAX GPU compute failed. cuDNN likely still missing."
    return 1 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Step 5: Install EXPO dependencies (all versions cross-verified)
# ---------------------------------------------------------------------------
echo "[5/6] Installing EXPO dependencies..."

# Core ML stack — EVERY version pinned to prevent pip from upgrading JAX.
# CRITICAL: numpy<2.0 because tensorflow-probability 0.19.0 uses np.issctype
#           (removed in NumPy 2.0). numpy>=1.24 satisfies JAX's requirement.
# CRITICAL: flax>=0.8.0 needed (0.7.5 uses define_bool_state removed in JAX 0.4.25+)
#           flax<=0.10.4 needed (0.10.5+ requires jax>=0.5.1)
# CRITICAL: distrax==0.1.5 because 0.1.6+ requires jax>=0.7.0
# CRITICAL: jax==0.4.35 re-stated so flax/chex can't upgrade it
"$PIP" install \
    "numpy>=1.24,<2.0" \
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

# System deps for mujoco_py compilation (GL/osmesa.h required)
echo "       Installing mujoco_py build deps (apt-get update + install)..."
apt-get update -qq
# Install each package separately — libgl1-mesa-glx is obsolete on newer Ubuntu
# and would cause the entire apt-get to fail if bundled together.
apt-get install -y libosmesa6-dev || echo "       WARN: libosmesa6-dev failed"
apt-get install -y libglew-dev    || echo "       WARN: libglew-dev failed"
apt-get install -y patchelf       || echo "       WARN: patchelf failed"
apt-get install -y ffmpeg         || echo "       WARN: ffmpeg failed"
apt-get install -y libgl1-mesa-dri 2>/dev/null || \
    apt-get install -y libgl1-mesa-glx 2>/dev/null || \
    echo "       WARN: mesa GL runtime not found (may already be installed)"

# Verify GL/osmesa.h exists — mujoco_py won't compile without it
if [ ! -f /usr/include/GL/osmesa.h ]; then
    echo "       WARNING: GL/osmesa.h not at /usr/include/GL/osmesa.h"
    OSMESA_PATH=$(find / -name osmesa.h -type f 2>/dev/null | head -1)
    if [ -n "$OSMESA_PATH" ]; then
        OSMESA_DIR=$(dirname "$OSMESA_PATH")
        echo "       Found osmesa.h at $OSMESA_PATH — adding $OSMESA_DIR to CPATH"
        export CPATH="$OSMESA_DIR${CPATH:+:$CPATH}"
    else
        echo "       ERROR: osmesa.h not found anywhere. mujoco_py will fail to compile."
        return 1 2>/dev/null || true
    fi
fi
echo "       GL/osmesa.h: OK"

"$PIP" install "Cython<3" "mujoco_py==2.1.2.14"

# dmcgym: --no-deps to avoid gym[mujoco] pulling legacy mujoco_py again
"$PIP" install --no-deps dmcgym@git+https://github.com/ikostrikov/dmcgym

# D4RL: --no-deps to avoid pulling pybullet (mujoco_py is now installed)
"$PIP" install --no-deps \
    d4rl@git+https://github.com/Farama-Foundation/D4RL.git@2b96431a0e9fd90c8032624b0dc3cd4514d15632

# D4RL's actual runtime deps (that --no-deps skipped)
"$PIP" install h5py click termcolor

# Adroit hand envs with binary rewards (pen-binary-v0, door-binary-v0, etc.)
# Standard mj_envs only has dense rewards. Binary variants are in the Cal-QL fork.
# Dependency chain: pen-binary-v0 → mj_envs (nakamotoo fork) → mjrl
"$PIP" install mjrl@git+https://github.com/aravindr93/mjrl.git
"$PIP" install mj_envs@git+https://github.com/nakamotoo/mj_envs.git

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

import numpy as np
print(f'NumPy:          {np.__version__}')

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
