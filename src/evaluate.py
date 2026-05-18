"""
evaluate.py
-----------
Evaluation utilities for the pharmacy ticket classification system.

Goes beyond simple accuracy — uses metrics that matter in clinical contexts:
  - Macro F1 (treats all classes equally, important for imbalanced data)
  - Per-class precision/recall (critical for identifying weak spots)
  - Confusion matrix (where does the model fail?)
  - CRITICAL-class recall (missing a CRITICAL ticket is a safety failure)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ─────────────────────────────────────────────
# CORE EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(pipeline, X_test, y_test) -> dict:
    """
    Evaluate a trained pipeline on test data.

    Returns a metrics dict with accuracy, per-class scores,
    macro averages, and the raw confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    labels = sorted(list(set(y_test)))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":          round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":       round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro":   round(precision_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":      round(recall_score(y_test, y_pred, average="macro",       zero_division=0), 4),
        "per_class":         {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"],    4),
                "f1":        round(report[label]["f1-score"],  4),
                "support":   int(report[label]["support"]),
            }
            for label in labels if label in report
        },
        "confusion_matrix":  confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels":            labels,
    }


def critical_class_recall(pipeline, X_test, y_test) -> float:
    """
    Compute recall specifically for the 'CRITICAL' priority class.

    In a healthcare triage system, failing to catch a CRITICAL ticket
    is a patient safety failure — this metric is more important than
    overall accuracy for the priority classifier.

    Returns
    -------
    float : Recall for CRITICAL class, or None if class not present.
    """
    if "CRITICAL" not in set(y_test):
        return None

    y_pred = pipeline.predict(X_test)
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    idx = labels.index("CRITICAL")
    tp = cm[idx, idx]
    fn = cm[idx].sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(recall, 4)


def print_report(metrics: dict, label: str = "Model"):
    """Print a formatted evaluation summary."""
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro        : {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted     : {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro : {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro    : {metrics['recall_macro']:.4f}")

    if "cv_f1_mean" in metrics:
        print(f"  CV F1 (5-fold)  : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

    print(f"\n  Per-class breakdown:")
    for cls, scores in metrics.get("per_class", {}).items():
        print(f"    {cls:<40} P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}  n={scores['support']}")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    metrics:    dict,
    task:       str = "Category",
    output_dir: str = "../reports/figures",
    show:       bool = False,
):
    """
    Plot and save a normalized confusion matrix heatmap.

    Parameters
    ----------
    metrics    : Output from evaluate_model()
    task       : 'Category' or 'Priority' — used for title and filename
    output_dir : Directory to save the figure
    show       : Display the plot interactively
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    labels = metrics["labels"]
    cm     = np.array(metrics["confusion_matrix"])

    # Normalize by row (true label) to show recall per class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 8))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Recall (row-normalized)"},
        linewidths=0.5,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title(f"Pharmacy Ticket {task} Classifier — Confusion Matrix\n"
                 f"(normalized by true class, diagonal = recall per class)",
                 fontsize=13, pad=15)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{task.lower()}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Confusion matrix saved: {out_path}")

    if show:
        plt.show()
    plt.close()


def plot_model_comparison(
    results_dict: dict,
    task:         str = "Category",
    output_dir:   str = "../reports/figures",
    show:         bool = False,
):
    """
    Bar chart comparing all models by F1 Macro, Accuracy, Precision, Recall.

    Parameters
    ----------
    results_dict : {model_name: {"metrics": {...}}} from train_all_models()
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results_dict.keys())
    metrics_to_plot = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels   = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    colors          = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        values = [results_dict[m]["metrics"][metric] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Pharmacy Ticket {task} Classifier — Model Comparison", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{task.lower()}_model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Model comparison chart saved: {out_path}")

    if show:
        plt.show()
    plt.close()


def plot_class_performance(
    metrics:    dict,
    task:       str = "Category",
    output_dir: str = "../reports/figures",
    show:       bool = False,
):
    """
    Horizontal bar chart showing per-class F1 scores.
    Makes class-level weaknesses immediately visible.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    per_class = metrics["per_class"]
    classes   = list(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]
    support   = [per_class[c]["support"] for c in classes]

    # Sort by F1 score
    sorted_pairs = sorted(zip(f1_scores, classes, support), reverse=True)
    f1_scores, classes, support = zip(*sorted_pairs)

    colors = ["#2ecc71" if f >= 0.80 else "#e67e22" if f >= 0.65 else "#e74c3c"
              for f in f1_scores]

    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.6)))

    bars = ax.barh(classes, f1_scores, color=colors, alpha=0.85, edgecolor="white")

    for bar, score, n in zip(bars, f1_scores, support):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}  (n={n})", va="center", fontsize=9)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title(f"Per-Class F1 Score — {task} Classifier\n"
                 f"🟢 ≥0.80  🟠 0.65–0.80  🔴 <0.65", fontsize=12, pad=12)
    ax.axvline(x=0.80, color="green", linestyle="--", alpha=0.5, label="0.80 threshold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{task.lower()}_per_class_f1.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Per-class F1 chart saved: {out_path}")

    if show:
        plt.show()
    plt.close()
