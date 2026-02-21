#!/bin/bash
set -e

###############################################################################
# EXPO Adaptive Beta — RunPod Setup Script
#
# Usage:
#   1. Create a RunPod pod with RTX 4090 + PyTorch 2.x template
#   2. SSH in and run:
#      curl -sSL <raw_url_of_this_script> | bash
#      OR
#      git clone <your-repo> && cd expo-adaptive-beta/EXPO && bash scripts/setup.sh
###############################################################################

echo "============================================"
echo "  EXPO Adaptive Beta — Environment Setup"
echo "============================================"

WORKDIR="${EXPO_WORKDIR:-/workspace/EXPO}"

# ---------------------------------------------------------------------------
# Step 1: Clone EXPO if not already present
# ---------------------------------------------------------------------------
if [ ! -f "$WORKDIR/train_finetuning.py" ]; then
    echo "[1/5] Cloning EXPO repo..."
    git clone https://github.com/pd-perry/EXPO.git "$WORKDIR"
    echo "       NOTE: You need to copy the modified files (edit_distance.py, expo_learner.py,"
    echo "       expo_config.py, plot_results.py) into this clone manually,"
    echo "       OR clone your fork instead."
else
    echo "[1/5] EXPO already present at $WORKDIR, skipping clone."
fi

cd "$WORKDIR"

# ---------------------------------------------------------------------------
# Step 2: Create Python venv (avoids conda, faster on RunPod)
# ---------------------------------------------------------------------------
echo "[2/5] Creating Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Step 3: Install JAX with CUDA support
# ---------------------------------------------------------------------------
echo "[3/5] Installing JAX with CUDA..."

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
    echo "       Detected CUDA $CUDA_VER"
else
    CUDA_VER="12"
    echo "       nvcc not found, assuming CUDA 12.x"
fi

# JAX 0.4.x with CUDA (matching environment.yml)
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---------------------------------------------------------------------------
# Step 4: Install EXPO dependencies
# ---------------------------------------------------------------------------
echo "[4/5] Installing EXPO dependencies..."

# Core ML
pip install \
    flax \
    "optax==0.1.5" \
    chex \
    distrax \
    ml_collections \
    orbax-checkpoint==0.2.3

# RL environment
pip install \
    "gym==0.23.1" \
    "gym[mujoco]" \
    mujoco \
    dm_control \
    dmcgym@git+https://github.com/ikostrikov/dmcgym

# D4RL (pinned commit from EXPO repo)
pip install \
    d4rl@git+https://github.com/Farama-Foundation/D4RL.git@2b96431a0e9fd90c8032624b0dc3cd4514d15632

# Logging and utils
pip install \
    wandb \
    absl-py \
    tqdm \
    numpy \
    scipy \
    h5py \
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
# Step 5: Verify installation
# ---------------------------------------------------------------------------
echo "[5/5] Verifying installation..."

python3 -c "
import jax
print(f'JAX version:    {jax.__version__}')
print(f'JAX devices:    {jax.devices()}')
print(f'GPU available:  {len(jax.devices(\"gpu\")) > 0}')

import flax
print(f'Flax version:   {flax.__version__}')

import optax
print(f'Optax version:  {optax.__version__}')

import gym
print(f'Gym version:    {gym.__version__}')

try:
    import d4rl
    print('D4RL:           OK')
except Exception as e:
    print(f'D4RL:           WARNING - {e}')

try:
    from expo.agents.sac.edit_distance import EditDistance
    print('EditDistance:    OK (adaptive beta module found)')
except ImportError:
    print('EditDistance:    NOT FOUND — copy modified files into this repo')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate env:  source $WORKDIR/venv/bin/activate"
echo "  Run experiments: bash scripts/run_all.sh"
echo "  Monitor:        watch nvidia-smi"
echo "  Wandb dashboard: https://wandb.ai"
echo "============================================"
