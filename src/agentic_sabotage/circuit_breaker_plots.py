"""Publication figures for Study 2.5: Closed-Loop Probe-Gated Circuit Breaker (`2B` & `9B`)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plotting import set_pub_style


def plot_polysemantic_cliff_rescue_2b(report: dict[str, Any], out_path: Path) -> None:
    """Figure 1: Rescuing Gemma-2-2B-IT's Polysemantic Over-Refusal Cliff via Probe-Gating."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.3))

    conds_2b = {c["method"]: c for c in report["direct_boundary_2b"]["conditions"]}
    ordered = [
        ("Unsteered\n(α = 0.00)", "unsteered", "#64748b"),
        ("Always-On\nDense (+0.20)", "always_on_dense_probe_a20", "#f97316"),
        ("Always-On\nDense (+0.25)", "always_on_dense_probe_a25", "#dc2626"),
        ("Always-On\nSAE (+0.25)", "always_on_sae_steer_a25", "#8b5cf6"),
        ("Probe-Gated\nDense Breaker", "probe_gated_dense_circuit_breaker", "#0284c7"),
        ("SAE-Gated\nSAE Breaker", "sae_gated_sae_circuit_breaker", "#059669"),
    ]
    ordered = [(lbl, k, col) for lbl, k, col in ordered if k in conds_2b]

    x = np.arange(len(ordered))
    w = 0.36
    sab_vals = [100.0 * conds_2b[k]["sabotage_rate_pressured"] for _, k, _ in ordered]
    ben_vals = [100.0 * conds_2b[k]["legit_task_pass_rate"] for _, k, _ in ordered]

    b1 = ax1.bar(x - w / 2, sab_vals, width=w, color="#dc2626", alpha=0.88, label="Pressured Sabotage Rate (↓ Better)")
    b2 = ax1.bar(x + w / 2, ben_vals, width=w, color="#059669", alpha=0.88, label="Benign Coding Accuracy (↑ Better)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.3,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for lbl, _, _ in ordered], fontsize=8.3)
    ax1.set_ylim(0, 124)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("(A) Gemma-2-2B-IT: Solving the Polysemantic Cliff (α = +0.25)")
    ax1.legend(loc="upper right", fontsize=8.3, frameon=True)

    # Panel B: Gate Selectivity (Trigger Rate on Pressured Sabotage Prompts vs. Benign Controls) across 2B & 9B
    conds_9b = {c["method"]: c for c in report["direct_boundary_9b"]["conditions"]}
    gate_Specs = [
        ("2B L16\nResid Gate", conds_2b.get("probe_gated_dense_circuit_breaker", {})),
        ("2B L16\nSAE Gate", conds_2b.get("sae_gated_sae_circuit_breaker", {})),
        ("9B L24\nResid Gate", conds_9b.get("probe_gated_dense_circuit_breaker", {})),
        ("9B L24\nSAE Gate", conds_9b.get("sae_gated_sae_circuit_breaker", {})),
    ]
    gx = np.arange(len(gate_Specs))
    trig_p = [100.0 * g.get("gate_trigger_rate_pressured", 0.0) for _, g in gate_Specs]
    trig_b = [100.0 * g.get("gate_trigger_rate_benign", 0.0) for _, g in gate_Specs]

    gb1 = ax2.bar(gx - w / 2, trig_p, width=w, color="#7c3aed", label="Gate Trigger Rate on Pressured Prompts (%)")
    gb2 = ax2.bar(gx + w / 2, trig_b, width=w, color="#0284c7", label="False-Trigger Rate on Benign Coding Controls (%)")

    for bars in (gb1, gb2):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.6,
                fontweight="bold",
            )

    ax2.set_xticks(gx)
    ax2.set_xticklabels([lbl for lbl, _ in gate_Specs])
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Circuit-Breaker Activation Rate (%)")
    ax2.set_title("(B) Selective Intervention: Near-Zero Triggering on Benign Tasks")
    ax2.legend(loc="upper right", fontsize=8.3, frameon=True)

    fig.suptitle("Closed-Loop Probe-Gated Circuit Breaker Eliminates Always-On Steering Collateral", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_closed_loop_cot_circuit_breaker_9b(report: dict[str, Any], out_path: Path) -> None:
    """Figure 2: Defeating `<think>` Action Re-Locking on Gemma-2-9B-IT via Streaming Silent-Commitment Latch."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.3))

    modes = {m["mode"]: m for m in report["cot_streaming_9b"]["modes"]}
    spec_rows = [
        ("Unsteered\n<think>+Tool", "unsteered_cot"),
        ("Steer <think>\nOnly (Re-Lock)", "steer_think_only_sae"),
        ("Post-<think>\nBoundary Gate", "post_think_boundary_gate_sae"),
        ("Streaming\nSAE Breaker", "streaming_cot_sae_circuit_breaker"),
        ("Streaming\nDense Breaker", "streaming_cot_dense_circuit_breaker"),
        ("Always-On\nFull SAE", "always_on_full_sae"),
    ]
    spec_rows = [(lbl, k) for lbl, k in spec_rows if k in modes]

    x = np.arange(len(spec_rows))
    w = 0.36
    sab = [100.0 * modes[k]["sabotage_rate"] for _, k in spec_rows]
    hon = [100.0 * modes[k]["honest_rate"] for _, k in spec_rows]

    b1 = ax1.bar(x - w / 2, sab, width=w, color="#dc2626", label="Downstream Sabotage Tool Call (%)")
    b2 = ax1.bar(x + w / 2, hon, width=w, color="#2563eb", label="Honest Escalation / Root-Cause Fix (%)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for lbl, _ in spec_rows], fontsize=8.3)
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Outcome Rate on Held-Out Focus Set (%)")
    ax1.set_title("(A) Defeating Action Re-Locking via Early-<think> Latched Steering")
    ax1.legend(loc="upper right", fontsize=8.2, frameon=True)

    # Panel B: Benign `<think>` Coding Fidelity vs. Intervention Overhead on Benign Traffic
    ben_rows = [
        ("Unsteered", "unsteered_cot"),
        ("Steer <think>\nOnly", "steer_think_only_sae"),
        ("Streaming SAE\nCircuit Breaker", "streaming_cot_sae_circuit_breaker"),
        ("Streaming Dense\nCircuit Breaker", "streaming_cot_dense_circuit_breaker"),
        ("Always-On\nFull Steer", "always_on_full_sae"),
    ]
    ben_rows = [(lbl, k) for lbl, k in ben_rows if k in modes]
    bx = np.arange(len(ben_rows))
    ben_acc = [100.0 * modes[k]["benign_cot_pass_rate"] for _, k in ben_rows]
    ben_int = [100.0 * modes[k]["gate_trigger_rate_benign"] for _, k in ben_rows]

    bb1 = ax2.bar(bx - w / 2, ben_acc, width=w, color="#059669", label="Benign <think> Coding Accuracy (%)")
    bb2 = ax2.bar(bx + w / 2, ben_int, width=w, color="#f59e0b", label="Hooks Activated on Benign Rollouts (%)")

    for bars in (bb1, bb2):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.6,
                fontweight="bold",
            )

    ax2.set_xticks(bx)
    ax2.set_xticklabels([lbl for lbl, _ in ben_rows], fontsize=8.5)
    ax2.set_ylim(0, 126)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(B) Zero-Overhead Unsteered Execution on Benign CoT Tasks")
    ax2.legend(loc="upper right", fontsize=8.2, frameon=True)

    fig.suptitle("Gemma-2-9B-IT Closed-Loop <think> Circuit Breaker: Early Detection Latched Across Action", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_breaker_suite(report: dict[str, Any], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        fig_dir / "1_polysemantic_cliff_rescue_2b.png",
        fig_dir / "2_closed_loop_cot_circuit_breaker_9b.png",
    ]
    plot_polysemantic_cliff_rescue_2b(report, outputs[0])
    plot_closed_loop_cot_circuit_breaker_9b(report, outputs[1])
    return outputs
