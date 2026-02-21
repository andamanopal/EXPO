"""Plot adaptive beta experiment results from wandb logs.

Generates three figures:
1. Beta trajectory — shows adaptive beta converging from init toward task-optimal
2. Learning curves — compares fixed vs adaptive beta performance
3. Edit magnitude tracking — shows self-regulating behavior

Usage:
    python scripts/plot_results.py --wandb_project expo --output_dir plots/adaptive_beta
    python scripts/plot_results.py --csv_dir logs/ --output_dir plots/adaptive_beta
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TASK_COLORS = {
    "antmaze": "#2196F3",
    "pen": "#FF5722",
}

STYLE_MAP = {
    "fixed_optimal": {"linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "adaptive_optimal_init": {"linestyle": "--", "linewidth": 1.8, "alpha": 0.8},
    "adaptive_wrong_init": {"linestyle": "-.", "linewidth": 1.8, "alpha": 0.8},
}


def smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_wandb_runs(project_name, entity=None):
    try:
        import wandb
    except ImportError as err:
        raise ImportError("Install wandb: pip install wandb") from err

    api = wandb.Api()
    project_path = f"{entity}/{project_name}" if entity else project_name
    return api.runs(project_path)


def load_csv_data(csv_dir):
    """Load data from CSV files exported from wandb.

    Expected structure:
        csv_dir/
            antmaze_fixed_optimal.csv
            antmaze_adaptive_wrong_init.csv
            pen_fixed_optimal.csv
            pen_adaptive_wrong_init.csv
    """
    data = {}
    csv_path = Path(csv_dir)
    for csv_file in csv_path.glob("*.csv"):
        name = csv_file.stem
        try:
            raw = np.genfromtxt(csv_file, delimiter=",", names=True)
            data[name] = raw
        except Exception as e:
            print(f"Warning: could not load {csv_file}: {e}")
    return data


def plot_beta_trajectory(runs_data, output_dir, optimal_betas=None):
    """Plot beta values over training steps."""
    fig, ax = plt.subplots()

    if optimal_betas is None:
        optimal_betas = {"antmaze": 0.05, "pen": 0.7}

    for task_name, task_data in runs_data.items():
        color = TASK_COLORS.get(task_name.split("_")[0], "#666666")
        steps = task_data.get("steps", np.arange(len(task_data["beta"])))
        beta_vals = smooth(task_data["beta"], window=50)
        steps_smooth = steps[: len(beta_vals)]

        ax.plot(steps_smooth, beta_vals, color=color, linewidth=2, label=task_name)

    for task_key, opt_val in optimal_betas.items():
        color = TASK_COLORS.get(task_key, "#666666")
        ax.axhline(
            y=opt_val,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            label=f"{task_key} optimal ({opt_val})",
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Beta (Edit Action Scale)")
    ax.set_title("Adaptive Beta Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "beta_trajectory.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_learning_curves(runs_data, output_dir):
    """Plot normalized returns comparing fixed vs adaptive beta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    task_groups = {}
    for run_name, run_data in runs_data.items():
        parts = run_name.split("_", 1)
        task = parts[0]
        variant = parts[1] if len(parts) > 1 else "unknown"
        task_groups.setdefault(task, {})[variant] = run_data

    for idx, (task, variants) in enumerate(sorted(task_groups.items())):
        ax = axes[idx] if len(task_groups) > 1 else axes
        color = TASK_COLORS.get(task, "#666666")

        for variant_name, variant_data in sorted(variants.items()):
            style = STYLE_MAP.get(variant_name, STYLE_MAP["fixed_optimal"])
            returns = variant_data.get("returns", variant_data.get("return", []))
            steps = variant_data.get("steps", np.arange(len(returns)))
            returns_smooth = smooth(np.array(returns), window=20)
            steps_smooth = steps[: len(returns_smooth)]

            label = variant_name.replace("_", " ").title()
            ax.plot(steps_smooth, returns_smooth, color=color, label=label, **style)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Normalized Return")
        ax.set_title(f"{task.title()} Learning Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_edit_magnitude(runs_data, output_dir):
    """Plot edit magnitude over training to show self-regulating behavior."""
    fig, ax = plt.subplots()

    for task_name, task_data in runs_data.items():
        color = TASK_COLORS.get(task_name.split("_")[0], "#666666")
        edit_mag = task_data.get("raw_edit_mag", task_data.get("edit_mag", []))
        steps = task_data.get("steps", np.arange(len(edit_mag)))
        edit_smooth = smooth(np.array(edit_mag), window=50)
        steps_smooth = steps[: len(edit_smooth)]

        ax.plot(steps_smooth, edit_smooth, color=color, linewidth=2, label=task_name)

    ax.axhline(
        y=0.5,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label="Target (0.5)",
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Raw Edit Magnitude")
    ax.set_title("Edit Magnitude Tracking (Self-Regulating Behavior)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "edit_magnitude.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_synthetic_demo(output_dir):
    """Generate demo plots with synthetic data to verify plot layout."""
    np.random.seed(42)
    steps = np.arange(0, 100000, 100)

    antmaze_beta = 0.3 * np.exp(-steps / 30000) + 0.05 * (1 - np.exp(-steps / 30000))
    antmaze_beta += np.random.normal(0, 0.005, len(steps))

    pen_beta = 0.3 + (0.7 - 0.3) * (1 - np.exp(-steps / 25000))
    pen_beta += np.random.normal(0, 0.01, len(steps))

    runs_data = {
        "antmaze_adaptive_wrong_init": {
            "steps": steps,
            "beta": antmaze_beta,
            "returns": np.clip(
                0.1 + 0.6 * (1 - np.exp(-steps / 40000))
                + np.random.normal(0, 0.05, len(steps)),
                0,
                1,
            ),
            "raw_edit_mag": np.clip(
                0.55 * np.exp(-steps / 20000)
                + 0.5 * (1 - np.exp(-steps / 20000))
                + np.random.normal(0, 0.02, len(steps)),
                0.1,
                0.9,
            ),
        },
        "pen_adaptive_wrong_init": {
            "steps": steps,
            "beta": pen_beta,
            "returns": np.clip(
                0.05 + 0.5 * (1 - np.exp(-steps / 50000))
                + np.random.normal(0, 0.03, len(steps)),
                0,
                1,
            ),
            "raw_edit_mag": np.clip(
                0.45 + 0.05 * np.sin(steps / 10000)
                + np.random.normal(0, 0.02, len(steps)),
                0.1,
                0.9,
            ),
        },
    }

    plot_beta_trajectory(runs_data, output_dir)
    plot_learning_curves(runs_data, output_dir)
    plot_edit_magnitude(runs_data, output_dir)
    print(f"\nDemo plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot adaptive beta results")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity")
    parser.add_argument("--csv_dir", type=str, help="Directory with CSV exports")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/adaptive_beta",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo plots with synthetic data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo:
        generate_synthetic_demo(args.output_dir)
        return

    if args.csv_dir:
        data = load_csv_data(args.csv_dir)
        plot_beta_trajectory(data, args.output_dir)
        plot_learning_curves(data, args.output_dir)
        plot_edit_magnitude(data, args.output_dir)
    elif args.wandb_project:
        print("Loading runs from wandb...")
        runs = load_wandb_runs(args.wandb_project, args.wandb_entity)
        print(f"Found {len(runs)} runs. Processing...")
        print("Use --csv_dir with exported CSVs or --demo for synthetic plots.")
    else:
        print("No data source specified. Use --demo for synthetic plots,")
        print("--csv_dir for CSV data, or --wandb_project for wandb.")


if __name__ == "__main__":
    main()
