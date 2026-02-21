# EXPO Adaptive Beta — Quickstart

## RunPod Setup (RTX 4090, ~5 min)

```bash
git clone -b adaptive-beta https://github.com/andamanopal/EXPO.git /workspace/EXPO
cd /workspace/EXPO
bash scripts/setup.sh
source venv/bin/activate
wandb login
```

## Validate (2 min)

```bash
bash scripts/run_all.sh --quick
```

Check logs if anything fails: `tail -f logs/*.log`

## Run Full Experiments

```bash
bash scripts/run_all.sh
```

Launches 4 parallel runs on one GPU:

| # | Experiment | Env | Beta | Steps | Time |
|---|-----------|-----|------|-------|------|
| 1 | Baseline (fixed optimal) | antmaze-large-diverse | 0.05 | 300K | ~1 hr |
| 2 | Baseline (fixed optimal) | pen-binary | 0.70 | 1M | ~3 hr |
| 3 | **Adaptive (wrong init)** | antmaze-large-diverse | 0.3 → ? | 300K | ~1 hr |
| 4 | **Adaptive (wrong init)** | pen-binary | 0.3 → ? | 1M | ~3 hr |

Runs 3 & 4 are the money shot — beta should converge from 0.3 toward the task-optimal values.

## Monitor

```bash
tail -f logs/adaptive_antmaze.log    # live output
watch nvidia-smi                      # GPU usage
# wandb dashboard for real-time curves
```

## Generate Plots (after experiments finish)

```bash
python scripts/plot_results.py --wandb_project expo-adaptive-beta --output_dir plots/results
```

Produces 3 figures:
1. **Beta trajectory** — beta converging from 0.3 toward optimal
2. **Learning curves** — fixed vs adaptive performance
3. **Edit magnitude** — self-regulating behavior

## What Success Looks Like

- antmaze adaptive beta: 0.3 → decreases toward ~0.05
- pen adaptive beta: 0.3 → increases toward ~0.7
- Adaptive runs approach fixed-optimal return (within ~80%)
- Edit magnitude stabilizes around 0.5 (the target)

## If Things Go Wrong

| Problem | Fix |
|---------|-----|
| OOM | Shouldn't happen (~3GB/run). Lower `--config.N=4` if needed |
| Beta stuck at 0.01 or 1.0 | Hit clip boundary. Try `--config.beta_lr=1e-4` |
| Bad returns | Start with `--max_steps=100000` to debug faster |
| Import error | Check `python -c "from expo.agents.sac.edit_distance import EditDistance"` |

## Key Files

```
configs/expo_config.py          — adaptive_beta, beta_lr, target_edit_mag
expo/agents/sac/edit_distance.py — EditDistance module (15 lines)
expo/agents/sac/expo_learner.py  — get_beta(), update_beta(), core logic
scripts/run_all.sh               — launches all 4 experiments
scripts/plot_results.py          — generates figures
```
