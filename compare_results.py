"""
compare_results.py - Compare Focal+Dice vs Weighted CE Training Results
Brain Tumor Segmentation - BRISC 2025 Dataset

Usage:
    python Code/compare_results.py

Reads test_results.json and training_log.csv from:
    outputs/baseline_focal_dice/
    outputs/baseline_ce/

Generates:
    outputs/comparison/comparison_table.txt
    outputs/comparison/comparison_curves.png
    outputs/comparison/comparison_bar_chart.png
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths to the two experiment directories
FOCAL_DIR = "outputs/baseline_focal_dice"
CE_DIR = "outputs/baseline_ce"
OUT_DIR = "outputs/comparison"

CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}


def load_results(exp_dir):
    """Load test_results.json from an experiment directory."""
    path = os.path.join(exp_dir, "test_results.json")
    if not os.path.exists(path):
        print(f"  [ERROR] Not found: {path}")
        print(f"  Run the training first, then re-run this script.")
        return None
    with open(path) as f:
        return json.load(f)


def load_training_log(exp_dir):
    """Load training_log.csv as a pandas DataFrame."""
    path = os.path.join(exp_dir, "training_log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def print_comparison_table(focal, ce):
    """Print a side-by-side comparison of test metrics."""
    metrics = [
        ("Test Loss", "test_loss"),
        ("Pixel Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Mean Dice (all)", "mean_dice"),
        ("Mean Dice (tumor)", "mean_dice_tumor"),
        ("Mean IoU", "mean_iou"),
        ("ROC-AUC", "roc_auc"),
    ]

    lines = []
    sep = "=" * 65
    lines.append(sep)
    lines.append("  COMPARISON: Focal+Dice vs Weighted CrossEntropy")
    lines.append(sep)
    lines.append(f"  {'Metric':<25s} | {'Focal+Dice':>12s} | {'Weighted CE':>12s} | {'Winner':>10s}")
    lines.append(f"  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for label, key in metrics:
        v_focal = focal.get(key)
        v_ce = ce.get(key)

        if v_focal is None or v_ce is None:
            s_focal = "N/A" if v_focal is None else f"{v_focal:.4f}"
            s_ce = "N/A" if v_ce is None else f"{v_ce:.4f}"
            winner = ""
        else:
            s_focal = f"{v_focal:.4f}"
            s_ce = f"{v_ce:.4f}"
            # Lower is better for loss, higher is better for everything else
            if key == "test_loss":
                winner = "Focal+Dice" if v_focal < v_ce else "Weighted CE"
            else:
                winner = "Focal+Dice" if v_focal > v_ce else "Weighted CE"

        lines.append(f"  {label:<25s} | {s_focal:>12s} | {s_ce:>12s} | {winner:>10s}")

    # Per-class Dice
    lines.append("")
    lines.append(f"  {'Per-Class Dice':<25s} | {'Focal+Dice':>12s} | {'Weighted CE':>12s} | {'Winner':>10s}")
    lines.append(f"  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for c in range(4):
        name = CLASS_NAMES[c]
        d_focal = focal["per_class"].get(name, {}).get("dice", 0)
        d_ce = ce["per_class"].get(name, {}).get("dice", 0)
        winner = "Focal+Dice" if d_focal > d_ce else "Weighted CE"
        lines.append(f"  {name:<25s} | {d_focal:>12.4f} | {d_ce:>12.4f} | {winner:>10s}")

    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    return text


def plot_training_curves(focal_log, ce_log, output_path):
    """Overlay training curves from both experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Loss ---
    ax = axes[0, 0]
    if focal_log is not None:
        ax.plot(focal_log["epoch"], focal_log["train_loss"], "b-", label="Focal+Dice Train", alpha=0.8)
        ax.plot(focal_log["epoch"], focal_log["val_loss"], "b--", label="Focal+Dice Val", alpha=0.8)
    if ce_log is not None:
        ax.plot(ce_log["epoch"], ce_log["train_loss"], "r-", label="Weighted CE Train", alpha=0.8)
        ax.plot(ce_log["epoch"], ce_log["val_loss"], "r--", label="Weighted CE Val", alpha=0.8)
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Mean Dice ---
    ax = axes[0, 1]
    if focal_log is not None:
        ax.plot(focal_log["epoch"], focal_log["mean_dice"], "b-o", label="Focal+Dice", markersize=3)
    if ce_log is not None:
        ax.plot(ce_log["epoch"], ce_log["mean_dice"], "r-s", label="Weighted CE", markersize=3)
    ax.set_title("Mean Dice Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Tumor Dice ---
    ax = axes[1, 0]
    if focal_log is not None:
        ax.plot(focal_log["epoch"], focal_log["tumor_dice"], "b-o", label="Focal+Dice", markersize=3)
    if ce_log is not None:
        ax.plot(ce_log["epoch"], ce_log["tumor_dice"], "r-s", label="Weighted CE", markersize=3)
    ax.set_title("Tumor Dice Score (classes 1,2,3 only)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Tumor Dice")
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Macro F1 ---
    ax = axes[1, 1]
    if focal_log is not None:
        ax.plot(focal_log["epoch"], focal_log["macro_f1"], "b-o", label="Focal+Dice", markersize=3)
    if ce_log is not None:
        ax.plot(ce_log["epoch"], ce_log["macro_f1"], "r-s", label="Weighted CE", markersize=3)
    ax.set_title("Macro F1 Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Training Comparison: Focal+Dice vs Weighted CE",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved -> {output_path}")


def plot_bar_chart(focal, ce, output_path):
    """Bar chart comparing key test metrics side-by-side."""
    metrics = {
        "Accuracy": ("accuracy", False),
        "Macro F1": ("macro_f1", False),
        "Mean Dice": ("mean_dice", False),
        "Tumor Dice": ("mean_dice_tumor", False),
        "Mean IoU": ("mean_iou", False),
    }

    # Per-class Dice
    for c in range(1, 4):
        name = CLASS_NAMES[c]
        metrics[f"{name.capitalize()} Dice"] = (f"per_class_dice_{c}", True)

    labels = list(metrics.keys())
    focal_vals = []
    ce_vals = []

    for label, (key, is_perclass) in metrics.items():
        if is_perclass:
            c = int(key.split("_")[-1])
            name = CLASS_NAMES[c]
            focal_vals.append(focal["per_class"].get(name, {}).get("dice", 0))
            ce_vals.append(ce["per_class"].get(name, {}).get("dice", 0))
        else:
            focal_vals.append(focal.get(key, 0) or 0)
            ce_vals.append(ce.get(key, 0) or 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, focal_vals, width, label="Focal+Dice",
                   color="#2196F3", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, ce_vals, width, label="Weighted CE",
                   color="#FF5722", alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Metrics Comparison: Focal+Dice vs Weighted CE",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved -> {output_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n--- Loading results ---")
    focal = load_results(FOCAL_DIR)
    ce = load_results(CE_DIR)

    if focal is None or ce is None:
        print("\n  Cannot compare: need results from both experiments.")
        print(f"  Focal+Dice dir: {FOCAL_DIR} -> {'FOUND' if focal else 'MISSING'}")
        print(f"  Weighted CE dir: {CE_DIR} -> {'FOUND' if ce else 'MISSING'}")
        print("\n  Run both training scripts first:")
        print("    python Code/train.py          # Focal+Dice loss")
        print("    python Code/train_ce.py       # Weighted CE loss")
        return

    # --- Comparison Table ---
    print()
    table_text = print_comparison_table(focal, ce)
    table_path = os.path.join(OUT_DIR, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    print(f"\n  Table saved -> {table_path}")

    # --- Training Curves ---
    print("\n--- Generating plots ---")
    focal_log = load_training_log(FOCAL_DIR)
    ce_log = load_training_log(CE_DIR)
    plot_training_curves(focal_log, ce_log,
                         os.path.join(OUT_DIR, "comparison_curves.png"))

    # --- Bar Chart ---
    plot_bar_chart(focal, ce, os.path.join(OUT_DIR, "comparison_bar_chart.png"))

    print("\n" + "=" * 65)
    print(f"  All comparison outputs saved to: {OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
