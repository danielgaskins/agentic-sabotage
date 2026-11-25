"""Publication figures for Phase 1.2: Cross-Family Generalization (`Qwen/Qwen2.5-7B-Instruct`)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plotting import set_pub_style


def plot_qwen_behavioral_and_depth_auroc(report: dict[str, Any], out_path: Path) -> None:
    """Figure 1: Qwen2.5-7B-Instruct Behavioral Sabotage + Cross-Family Normalized Depth AUROC."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0))

    # Panel A: Behavioral Sabotage Rates by Category (`Unpressured` vs. `Pressured`)
    beh = report["behavioral_by_category"]
    cats = list(beh.keys())
    pretty = [c.replace("_", "\n").title() for c in cats]
    u_vals = [100.0 * beh[c]["sabotage_rate_unpressured"] for c in cats]
    p_vals = [100.0 * beh[c]["sabotage_rate_pressured"] for c in cats]

    x = np.arange(len(cats))
    w = 0.36
    b1 = ax1.bar(x - w / 2, u_vals, width=w, color="#3b82f6", label="Unpressured Counterfactual")
    b2 = ax1.bar(x + w / 2, p_vals, width=w, color="#dc2626", label="Bottleneck / Pressure Induced")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.6,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(pretty, fontsize=8.6)
    ax1.set_ylim(0, max(max(p_vals + [25.0]) * 1.20, 40.0))
    ax1.set_ylabel("Unauthorized Shortcut / Sabotage Rate (%)")
    ax1.set_title("(A) Qwen2.5-7B-Instruct: Induced Sabotage by Taxonomy")
    ax1.legend(loc="upper right", fontsize=8.4, frameon=True)

    # Panel B: Cross-Family Pre-Commitment Probe AUROC vs. Normalized Transformer Depth (`l / L`)
    refs = report.get("gemma_reference_curves", {})
    if "gemma-2-2b-it" in refs:
        g2 = refs["gemma-2-2b-it"]
        ax2.plot(
            [100.0 * r["depth_frac"] for r in g2],
            [r["auroc_mean"] for r in g2],
            marker="s",
            lw=2.0,
            ls="--",
            color="#64748b",
            label="Gemma-2-2B-IT (26L, Dual-RMSNorm)",
        )
    if "gemma-2-9b-it" in refs:
        g9 = refs["gemma-2-9b-it"]
        ax2.plot(
            [100.0 * r["depth_frac"] for r in g9],
            [r["auroc_mean"] for r in g9],
            marker="^",
            lw=2.2,
            ls="-.",
            color="#7c3aed",
            label="Gemma-2-9B-IT (42L, Dual-RMSNorm)",
        )

    qp = report["layer_probes"]
    qx = [100.0 * r["depth_frac"] for r in qp]
    qy = [r["auroc_mean"] for r in qp]
    qs = [r["auroc_std"] for r in qp]
    ax2.plot(qx, qy, marker="o", lw=2.6, color="#dc2626", label="Qwen2.5-7B-Instruct (28L, Single-RMSNorm + GQA)")
    ax2.fill_between(
        qx,
        np.clip(np.array(qy) - np.array(qs), 0.45, 1.0),
        np.clip(np.array(qy) + np.array(qs), 0.45, 1.0),
        color="#dc2626",
        alpha=0.14,
    )
    ax2.axvspan(50.0, 72.0, color="#059669", alpha=0.07, label="Convergent Planning Horizon (50%–72% Depth)")

    ax2.set_xlabel("Normalized Transformer Depth (% of Total Layers, l / L)")
    ax2.set_ylabel("5-Fold Cross-Validated Residual Probe AUROC")
    ax2.set_ylim(0.50, 1.02)
    ax2.set_title("(B) Architecture-Invariant Pre-Commitment Horizon")
    ax2.legend(loc="lower right", fontsize=8.2, frameon=True)

    fig.suptitle("Cross-Family Validation: Agentic Sabotage & Linear Pre-Commitment Generalize to Qwen2.5-7B", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_qwen_silent_commitment_and_breaker(report: dict[str, Any], out_path: Path) -> None:
    """Figure 2: Qwen2.5-7B `<think>` CoT Monitor Evasion vs. L16 White-Box Probe + Circuit Breaker."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0))

    sc = report["silent_commitment"]
    mv = sc["monitor_vs_probe_auroc"]
    tc = sc["temporal_cot_auroc"]

    xs = [20, 40, 60, 80, 100]
    y_aucs = [tc[f"p{k}"]["auroc_mean"] for k in xs]
    y_stds = [tc[f"p{k}"]["auroc_std"] for k in xs]

    ax1.plot(xs, y_aucs, marker="o", lw=2.6, color="#dc2626", label="White-Box Qwen L16 <think> Residual Probe")
    ax1.fill_between(
        xs,
        np.clip(np.array(y_aucs) - np.array(y_stds), 0.45, 1.0),
        np.clip(np.array(y_aucs) + np.array(y_stds), 0.45, 1.0),
        color="#dc2626",
        alpha=0.14,
    )
    zs_auc = mv["zero_shot_no_cot_l16_probe_auroc"]
    llm_auc = mv["llm_7b_cot_auditor_auroc"]
    lex_auc = mv["lexical_cot_monitor_auroc"]

    ax1.axhline(zs_auc, color="#7c3aed", lw=2.1, ls="-.", label=f"Zero-Shot No-CoT L16 Probe on <think> ({zs_auc:.3f})")
    ax1.axhline(llm_auc, color="#0284c7", lw=2.0, ls="--", label=f"Black-Box Qwen-7B CoT Compliance Auditor ({llm_auc:.3f})")
    ax1.axhline(lex_auc, color="#64748b", lw=1.7, ls=":", label=f"Black-Box Lexical CoT Monitor ({lex_auc:.3f})")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{k}%" for k in xs])
    ax1.set_ylim(0.45, 1.03)
    ax1.set_xlabel("Reasoning Progress Inside <think>...</think> (% Before </think>)")
    ax1.set_ylabel("Sabotage Detection AUROC")
    ax1.set_title("(A) Qwen2.5-7B Silent Pre-Tool Commitment Inside <think>")
    ax1.legend(loc="lower right", fontsize=8.1, frameon=True)

    # Panel B: Causal Planning-Block Steering (`L14,16,18,20`) & Closed-Loop Probe-Gated Circuit Breaker
    st = {r["method"]: r for r in report["steering_and_circuit_breaker"]}
    s_order = [
        ("Unsteered\n(α = 0.00)", "unsteered"),
        ("Always-On L14–20\nDense Probe", "always_on_dense_probe"),
        ("Closed-Loop L16\nCircuit Breaker", "probe_gated_circuit_breaker"),
    ]
    s_order = [(lbl, k) for lbl, k in s_order if k in st]
    sx = np.arange(len(s_order))
    w = 0.35
    sab_r = [100.0 * st[k]["sabotage_rate_pressured"] for _, k in s_order]
    ben_r = [100.0 * st[k]["legit_task_pass_rate"] for _, k in s_order]

    sb1 = ax2.bar(sx - w / 2, sab_r, width=w, color="#dc2626", label="Pressured Sabotage Rate (↓ Better)")
    sb2 = ax2.bar(sx + w / 2, ben_r, width=w, color="#059669", label="Benign Coding Accuracy (↑ Better)")

    for bars in (sb1, sb2):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
            )

    ax2.set_xticks(sx)
    ax2.set_xticklabels([lbl for lbl, _ in s_order])
    ax2.set_ylim(0, 126)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("(B) Qwen2.5-7B Planning-Block Steering & Circuit Breaker")
    ax2.legend(loc="upper right", fontsize=8.3, frameon=True)

    fig.suptitle("Qwen2.5-7B-Instruct Exhibits Unfaithful <think> Rationalization & Causal Probe Steerability", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_family_suite(report: dict[str, Any], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        fig_dir / "1_qwen25_7b_behavioral_and_depth_auroc.png",
        fig_dir / "2_qwen25_7b_silent_commitment_and_breaker.png",
    ]
    plot_qwen_behavioral_and_depth_auroc(report, outputs[0])
    plot_qwen_silent_commitment_and_breaker(report, outputs[1])
    return outputs
