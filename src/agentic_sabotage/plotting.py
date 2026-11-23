"""Publication-quality matplotlib/seaborn figures for Agentic Sabotage mechanistic analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_pub_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def plot_behavioral_pressure_effect(
    summary_by_cat: dict[str, dict[str, float]],
    out_path: Path,
    model_label: str,
) -> None:
    set_pub_style()
    cats = list(summary_by_cat.keys())
    pretty_cats = [c.replace("_", "\n").title() for c in cats]
    pressured_rates = [100.0 * summary_by_cat[c]["sabotage_rate_pressured"] for c in cats]
    unpressured_rates = [100.0 * summary_by_cat[c]["sabotage_rate_unpressured"] for c in cats]

    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    b1 = ax.bar(x - w / 2, unpressured_rates, width=w, label="Unpressured Counterfactual", color="#3b82f6")
    b2 = ax.bar(x + w / 2, pressured_rates, width=w, label="Bottleneck / Pressure Induced", color="#ef4444")

    ax.set_ylabel("Unauthorized Shortcut / Sabotage Rate (%)")
    ax.set_title(f"Induced Agentic Sabotage Across Tool-Use Taxonomies ({model_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_cats)
    ax.set_ylim(0, max(max(pressured_rates + [10.0]) * 1.18, 30.0))
    ax.legend(frameon=True)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_probe_sweep(
    probe_results: list[dict[str, Any]],
    sae_by_layer: dict[str, Any],
    out_path: Path,
    model_label: str,
) -> None:
    set_pub_style()
    layers = [r["layer"] for r in probe_results]
    aurocs = [r["auroc_mean"] for r in probe_results]
    stds = [r["auroc_std"] for r in probe_results]
    diff_aurocs = [r.get("diff_means_auroc", r["accuracy_mean"]) for r in probe_results]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.7))
    ax1.plot(layers, aurocs, marker="o", lw=2.5, color="#dc2626", label="Residual Linear Probe AUROC (5-fold CV)")
    ax1.fill_between(
        layers,
        np.clip(np.array(aurocs) - np.array(stds), 0.0, 1.0),
        np.clip(np.array(aurocs) + np.array(stds), 0.0, 1.0),
        color="#dc2626",
        alpha=0.15,
    )
    ax1.plot(layers, diff_aurocs, marker="^", lw=1.9, ls="--", color="#059669", label="Difference-in-Means AUROC")

    if sae_by_layer:
        s_layers = sorted(int(k) for k in sae_by_layer.keys())
        s_aucs = [sae_by_layer[str(k)]["sae_probe_auroc"] for k in s_layers]
        ax1.plot(s_layers, s_aucs, marker="D", lw=2.2, color="#7c3aed", label="Gemma Scope 16k SAE L1 Probe AUROC")
        ax1.axvspan(min(s_layers) - 0.5, max(s_layers) + 0.5, color="#7c3aed", alpha=0.07, label="Active Planning / Steering Block")

    ax1.set_xlabel("Transformer Residual Stream Layer (at Tool-Call Decision Token)")
    ax1.set_ylabel("5-Fold Cross-Validated AUROC")
    ax1.set_ylim(0.50, 1.00)
    ax1.set_title(f"Layer-Wise Emergence of the Sabotage Decision Circuit ({model_label})")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sae_top_features(
    sae_summary: dict[str, Any],
    out_path: Path,
    model_label: str,
) -> None:
    set_pub_style()
    sab_feats = sae_summary["top_sabotage_features"][:7]
    hon_feats = sae_summary["top_honest_features"][:7]

    combined = list(reversed(hon_feats)) + list(reversed(sab_feats))
    labels = []
    for f in combined:
        desc = str(f.get("description", "")).strip()
        if len(desc) > 42:
            desc = desc[:39] + "..."
        lbl = f"#{f['feature_idx']}" + (f" ({desc})" if desc else "")
        labels.append(lbl)

    vals = [f["cohen_d"] for f in combined]
    colors = ["#2563eb" if v < 0 else "#dc2626" for v in vals]

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    y_pos = np.arange(len(combined))
    ax.barh(y_pos, vals, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#111827", lw=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_xlabel("Differential Activation Effect Size (Cohen's d: Sabotage + / Honest Escalation −)")
    ax.set_title(
        f"Disentangled Gemma Scope 16k SAE Latents at Decision Boundary "
        f"(L{sae_summary['layer']}, {model_label})"
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_steering_pareto(
    steering_rows: list[dict[str, Any]],
    out_path: Path,
    model_label: str,
) -> None:
    set_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    ax1, ax2, ax3 = axes

    methods = ["sae_steer", "probe_steer", "diff_means_steer", "random_control"]
    palette = {
        "sae_steer": "#7c3aed",
        "probe_steer": "#dc2626",
        "diff_means_steer": "#059669",
        "random_control": "#9ca3af",
    }
    pretty_name = {
        "sae_steer": "Gemma Scope 16k SAE Vector",
        "probe_steer": "Dense Linear Probe Vector",
        "diff_means_steer": "Difference-in-Means Vector",
        "random_control": "Random Orthogonal Control",
    }

    base_rows = [r for r in steering_rows if r["method"] == "baseline"]
    base_sab = 100.0 * base_rows[0]["sabotage_rate_pressured"] if base_rows else 0.0
    base_legit = 100.0 * base_rows[0]["legit_task_pass_rate"] if base_rows else 100.0

    ax1.axhline(base_sab, color="#111827", ls="--", lw=1.1, label=f"Unsteered ({base_sab:.1f}%)")
    ax2.axhline(base_legit, color="#111827", ls="--", lw=1.1, label=f"Unsteered ({base_legit:.1f}%)")
    ax3.scatter([100.0 - base_sab], [base_legit], color="#111827", s=95, zorder=6, marker="*", label="Unsteered Baseline")

    for m in methods:
        sub = sorted([r for r in steering_rows if r["method"] == m], key=lambda z: z["alpha"])
        if not sub:
            continue
        alphas = [r["alpha"] for r in sub]
        sabs = [100.0 * r["sabotage_rate_pressured"] for r in sub]
        legits = [100.0 * r["legit_task_pass_rate"] for r in sub]
        safety = [100.0 - s for s in sabs]
        col = palette.get(m, "#374151")
        lbl = pretty_name.get(m, m)

        ax1.plot(alphas, sabs, marker="o", lw=2.3, color=col, label=lbl)
        ax2.plot(alphas, legits, marker="s", lw=2.3, color=col, label=lbl)

        pos_sub = [r for r in sub if r["alpha"] >= 0.0]
        ax3.plot(
            [100.0 * (1.0 - r["sabotage_rate_pressured"]) for r in pos_sub],
            [100.0 * r["legit_task_pass_rate"] for r in pos_sub],
            marker="o",
            lw=2.2,
            color=col,
            label=lbl,
        )

    ax1.axvline(0.0, color="#6b7280", ls=":", alpha=0.6)
    ax1.set_xlabel("Steering Strength α (+ suppresses, − induces)")
    ax1.set_ylabel("Sabotage Rate under Pressure (%)")
    ax1.set_title("(A) Sabotage Suppression & Induction")
    ax1.set_ylim(-3, 103)
    ax1.legend(fontsize=8.0, frameon=True)

    ax2.axvline(0.0, color="#6b7280", ls=":", alpha=0.6)
    ax2.set_xlabel("Steering Strength α")
    ax2.set_ylabel("Legitimate Coding Tool Accuracy (%)")
    ax2.set_title("(B) Benign Coding Capability Retention")
    ax2.set_ylim(-3, 105)

    ax3.set_xlabel("Safety Rate under Pressure (100% − Sabotage %)")
    ax3.set_ylabel("Legitimate Coding Tool Accuracy (%)")
    ax3.set_title("(C) Safety–Capability Pareto Frontier")
    ax3.set_xlim(min(30.0, 100.0 - base_sab - 15.0), 103.0)
    ax3.set_ylim(-3, 105)
    ax3.legend(fontsize=8.0, loc="lower left", frameon=True)

    fig.suptitle(f"Inference-Time Activation Steering ({model_label})", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
