#!/bin/bash
set -e

###############################################################################
# EXPO Adaptive Beta — Run All Experiments (Sequential)
#
# Runs 4 experiments one at a time for easy debugging:
#   1. Baseline antmaze (fixed beta=0.05)
#   2. Baseline pen     (fixed beta=0.70)
#   3. Adaptive antmaze (init=0.3, should converge toward ~0.05)
#   4. Adaptive pen     (init=0.3, should converge toward ~0.70)
#
# Usage:
#   bash scripts/run_all.sh              # Full 300K/1M steps
#   bash scripts/run_all.sh --quick      # Quick 10K validation run
###############################################################################

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

PYTHON="$WORKDIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: .venv not found. Run '. scripts/setup.sh' first."
    exit 1
fi

# System CUDA paths (same as setup.sh — needed if run in a fresh shell)
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export PATH="$CUDA_ROOT/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=$CUDA_ROOT}"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export D4RL_SUPPRESS_IMPORT_ERROR=1

PROJECT="expo-adaptive-beta"
SEED=42

# Quick mode: short runs to validate code works
if [ "$1" = "--quick" ]; then
    ANTMAZE_STEPS=10000
    PEN_STEPS=10000
    EVAL_INTERVAL=2000
    echo "=== QUICK MODE: ${ANTMAZE_STEPS} steps per run ==="
else
    ANTMAZE_STEPS=300000
    PEN_STEPS=1000000
    EVAL_INTERVAL=5000
    echo "=== FULL MODE: antmaze=${ANTMAZE_STEPS}, pen=${PEN_STEPS} steps ==="
fi

# Common flags matching the paper's exact settings
COMMON_FLAGS="--config=configs/expo_config.py \
    --expo=True \
    --utd_ratio=20 \
    --config.backup_entropy=False \
    --config.hidden_dims=(256,256,256) \
    --config.num_min_qs=1 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --eval_interval=$EVAL_INTERVAL \
    --project_name=$PROJECT"

echo ""
echo "Wandb project: $PROJECT"
echo ""

# ---------------------------------------------------------------------------
# Experiment 1: Baseline antmaze — fixed optimal beta=0.05
# ---------------------------------------------------------------------------
echo "============================================"
echo "[1/4] baseline antmaze (fixed beta=0.05)"
echo "============================================"
"$PYTHON" train_finetuning.py \
    $COMMON_FLAGS \
    --env_name=antmaze-large-diverse-v2 \
    --seed=$SEED \
    --max_steps=$ANTMAZE_STEPS \
    --start_training=5000 \
    --config.edit_action_scale=0.05

# ---------------------------------------------------------------------------
# Experiment 2: Baseline pen — fixed optimal beta=0.70
# ---------------------------------------------------------------------------
echo "============================================"
echo "[2/4] baseline pen (fixed beta=0.70)"
echo "============================================"
"$PYTHON" train_finetuning.py \
    $COMMON_FLAGS \
    --env_name=pen-binary-v0 \
    --seed=$SEED \
    --max_steps=$PEN_STEPS \
    --start_training=0 \
    --config.edit_action_scale=0.7 \
    --config.actor_drop=0.1

# ---------------------------------------------------------------------------
# Experiment 3: Adaptive antmaze — wrong init beta=0.3
# ---------------------------------------------------------------------------
echo "============================================"
echo "[3/4] adaptive antmaze (init=0.3)"
echo "============================================"
"$PYTHON" train_finetuning.py \
    $COMMON_FLAGS \
    --env_name=antmaze-large-diverse-v2 \
    --seed=$SEED \
    --max_steps=$ANTMAZE_STEPS \
    --start_training=5000 \
    --config.edit_action_scale=0.3 \
    --config.adaptive_beta=True

# ---------------------------------------------------------------------------
# Experiment 4: Adaptive pen — wrong init beta=0.3
# ---------------------------------------------------------------------------
echo "============================================"
echo "[4/4] adaptive pen (init=0.3)"
echo "============================================"
"$PYTHON" train_finetuning.py \
    $COMMON_FLAGS \
    --env_name=pen-binary-v0 \
    --seed=$SEED \
    --max_steps=$PEN_STEPS \
    --start_training=0 \
    --config.edit_action_scale=0.3 \
    --config.adaptive_beta=True \
    --config.actor_drop=0.1

echo ""
echo "============================================"
echo "  All 4 experiments completed!"
echo ""
echo "  Generate plots:"
echo "    $PYTHON scripts/plot_results.py --wandb_project $PROJECT --output_dir plots/results"
echo "============================================"
