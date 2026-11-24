"""Publication-quality figures for Study 2: Silent Commitment (Unfaithful CoT vs. Layer-24 Probes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plotting import set_pub_style


def plot_temporal_commitment_auroc(report: dict[str, Any], out_path: Path) -> None:
    """Figure 1: Temporal emergence of sabotage commitment across `<think>` vs. Black-Box CoT Monitors."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))

    # Panel A: Normalized `<think>` progress (`20% -> 100%` strictly BEFORE `TOOL_CALL:`) + Tool Boundary
    tc = report["temporal_checkpoint_auroc"]
    mv = report["monitor_vs_probe_auroc"]
    xs = [20, 40, 60, 80, 100]
    pref_aucs = [tc[f"p{x}"]["prefix_mean_resid_auroc"] for x in xs]
    pref_stds = [tc[f"p{x}"]["prefix_mean_resid_std"] for x in xs]
    inst_aucs = [tc[f"p{x}"]["instantaneous_resid_auroc"] for x in xs]

    ax1.plot(xs, pref_aucs, marker="o", lw=2.6, color="#dc2626", label="White-Box L24 Residual Probe (Prefix Mean)")
    ax1.fill_between(
        xs,
        np.clip(np.array(pref_aucs) - np.array(pref_stds), 0.45, 1.0),
        np.clip(np.array(pref_aucs) + np.array(pref_stds), 0.45, 1.0),
        color="#dc2626",
        alpha=0.14,
    )
    ax1.plot(xs, inst_aucs, marker="s", lw=1.9, ls="--", color="#ef4444", alpha=0.8, label="White-Box L24 Residual Probe (Single Token)")

    sae_auc = mv["pretool_cot_sae_16k_l1_probe_auroc"]
    llm_auc = mv["llm_9b_cot_auditor_auroc"]
    lex_auc = mv["lexical_cot_monitor_auroc"]

    ax1.axhline(sae_auc, color="#7c3aed", lw=2.3, ls="-.", label=f"White-Box L24 Gemma Scope 16k SAE Probe ({sae_auc:.3f})")
    ax1.axhline(llm_auc, color="#0284c7", lw=2.1, ls=":", label=f"Black-Box 9B LLM CoT Compliance Auditor ({llm_auc:.3f})")
    ax1.axhline(lex_auc, color="#6b7280", lw=1.7, ls=":", label=f"Black-Box Lexical CoT Monitor ({lex_auc:.3f})")

    ax1.set_xlabel("Reasoning Progress Inside <think>...</think> (% Before </think>)")
    ax1.set_ylabel("5-Fold Cross-Validated AUROC")
    ax1.set_title("(A) Silent Pre-Tool Commitment Across <think> Scratchpad")
    ax1.set_ylim(0.45, 1.02)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{x}%" for x in xs])
    ax1.legend(loc="lower right", fontsize=8.2, frameon=True)

    # Panel B: Zero-Shot Study-1 `TOOL_CALL:` Probe transferred backward into `<think>` (`-48 .. -1` tokens)
    lb = report["lookback_zero_shot_auroc"]
    offsets = [48, 32, 20, 10, 5, 1]
    lb_vals = [lb[f"minus_{k}"] for k in offsets]
    x_labels = [f"-{k}" for k in offsets]
    x_idx = np.arange(len(offsets))

    bars = ax2.bar(x_idx, lb_vals, width=0.56, color="#7c3aed", alpha=0.86, label="Zero-Shot Study-1 L24 Probe (No CoT Retraining)")
    ax2.axhline(llm_auc, color="#0284c7", lw=2.0, ls="--", label=f"Full-<think> 9B CoT Text Auditor ({llm_auc:.3f})")
    ax2.axhline(0.50, color="#9ca3af", lw=1.0, ls=":")

    for bar, v in zip(bars, lb_vals):
        ax2.annotate(
            f"{v:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Token Offset Prior to </think> (Strictly Before TOOL_CALL:)")
    ax2.set_ylabel("Zero-Shot Sabotage Detection AUROC")
    ax2.set_title("(B) Zero-Shot Study-1 Probe Lead Time Before Action")
    ax2.set_ylim(0.45, 1.04)
    ax2.legend(loc="lower right", fontsize=8.4, frameon=True)

    fig.suptitle("Gemma-2-9B-IT Commits to Sabotage Inside <think> Before Verbalizing Action", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_unfaithful_cot_single_trace(report: dict[str, Any], out_path: Path) -> None:
    """Figure 2: Token-by-token Layer-24 SAE Latent ribbon (#11381, #15689, #6787) on an Unfaithful CoT trace."""
    set_pub_style()
    showcases = [s for s in report.get("showcase_silent_commitments", []) if s.get("cot_tokens")]
    if not showcases:
        return
    ex = showcases[0]
    tokens = ex["cot_tokens"]
    t_len = len(tokens)
    x = np.arange(t_len)

    lat_11381 = np.array(ex["latent_traces"]["11381"], dtype=np.float32)
    lat_15689 = np.array(ex["latent_traces"]["15689"], dtype=np.float32)
    lat_6787 = np.array(ex["latent_traces"]["6787"], dtype=np.float32)
    zs_probe = np.array(ex["zero_shot_probe_trace"], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.8, 6.4), gridspec_kw={"height_ratios": [2.2, 1.3]}, sharex=True)

    ax1.plot(x, lat_15689, lw=2.3, color="#dc2626", label="SAE Latent #15689 (Sabotage Spec / Protocol Bypass, d=+1.49)")
    ax1.fill_between(x, 0, lat_15689, color="#dc2626", alpha=0.14)
    ax1.plot(x, lat_11381, lw=2.4, color="#ea580c", label="SAE Latent #11381 (Causal Responsibility / Blame Shift, d=+1.04)")
    ax1.fill_between(x, 0, lat_11381, color="#ea580c", alpha=0.18)
    ax1.plot(x, lat_6787, lw=2.2, ls="--", color="#2563eb", label="SAE Latent #6787 (Honest Algorithmic Repair, d=-1.48)")

    ax1.set_ylabel("Layer-24 SAE JumpReLU Activation")
    cat_pretty = ex["category"].replace("_", " ").title()
    audit_pct = 100.0 * ex["llm_cot_auditor_score"]
    sae_pct = 100.0 * ex.get("pretool_sae_probe_prob", 0.9)
    ax1.set_title(
        f"Unfaithful <think> Trace ({cat_pretty}, {ex['id']}) — "
        f"External 9B CoT Auditor: {audit_pct:.0f}% Suspicion (FOOLED) vs. L24 SAE Probe: {sae_pct:.0f}% Sabotage"
    )
    ax1.legend(loc="upper left", fontsize=8.6, frameon=True)

    ax2.plot(x, zs_probe, lw=2.2, color="#7c3aed", label="Zero-Shot Study-1 L24 Sabotage Subspace Cosine")
    ax2.axhline(0.0, color="#6b7280", ls=":", lw=1.0)
    ax2.set_ylabel("L24 Probe Cosine")
    ax2.set_xlabel("Token Position Inside <think>...</think> Scratchpad (Prior to TOOL_CALL:)")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Sparse readable token ticks along X-axis
    step = max(1, t_len // 26)
    tick_idx = list(range(0, t_len, step))
    tick_lbls = [repr(tokens[i])[1:-1][:12] for i in tick_idx]
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_lbls, rotation=48, ha="right", fontsize=8.0)

    # Annotate snippet of benign-sounding CoT + actual destructive tool call at bottom
    cot_snip = " ".join(ex["cot_text"].split())[:175]
    tool_snip = ex["tool_call"][:135]
    fig.text(
        0.01,
        -0.04,
        f'Benign-Sounding <think>: "{cot_snip}..."\nActual Emitted Action:   {tool_snip}',
        fontsize=8.6,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_monitor_evasion_quadrant(report: dict[str, Any], out_path: Path) -> None:
    """Figure 3: Black-Box 9B CoT Auditor vs. White-Box Pre-Tool Layer-24 SAE Probe Quadrant + Evasion Rates."""
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))

    recs = [r for r in report["records"] if r["outcome"] in ("sabotage", "honest") and "pretool_sae_probe_prob" in r]
    hon = [r for r in recs if r["outcome"] == "honest"]
    sab_unmon = [r for r in recs if r["outcome"] == "sabotage" and r["regime"] == "unmonitored"]
    sab_audit = [r for r in recs if r["outcome"] == "sabotage" and r["regime"] == "audited"]

    # Highlight Silent Commitment Zone (X < 0.50 CoT Auditor Fooled, Y >= 0.50 SAE Probe Caught!)
    ax1.axvspan(-0.04, 0.50, ymin=0.50 / 1.08, ymax=1.0, color="#fef2f2", alpha=0.85, zorder=0)
    ax1.text(0.03, 0.93, "SILENT COMMITMENT ZONE\n(CoT Auditor Fooled,\nL24 SAE Probe Catches)", fontsize=8.5, fontweight="bold", color="#991b1b")

    ax1.scatter(
        [r["llm_cot_auditor_score"] for r in hon],
        [r["pretool_sae_probe_prob"] for r in hon],
        color="#2563eb",
        alpha=0.62,
        s=36,
        label=f"Honest Escalation / Root-Cause Fix (n={len(hon)})",
    )
    ax1.scatter(
        [r["llm_cot_auditor_score"] for r in sab_unmon],
        [r["pretool_sae_probe_prob"] for r in sab_unmon],
        color="#dc2626",
        alpha=0.78,
        s=44,
        marker="o",
        label=f"Sabotage — Unmonitored <think> (n={len(sab_unmon)})",
    )
    if sab_audit:
        ax1.scatter(
            [r["llm_cot_auditor_score"] for r in sab_audit],
            [r["pretool_sae_probe_prob"] for r in sab_audit],
            color="#7c3aed",
            alpha=0.85,
            s=52,
            marker="^",
            label=f"Sabotage — Audited <think> Notice (n={len(sab_audit)})",
        )

    ax1.axvline(0.50, color="#374151", ls="--", lw=1.1)
    ax1.axhline(0.50, color="#374151", ls="--", lw=1.1)
    ax1.set_xlim(-0.04, 1.04)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_xlabel("External Black-Box 9B CoT Auditor Suspicion Score")
    ax1.set_ylabel("Pre-Tool Layer-24 SAE Probe Probability (Inside <think>)")
    ax1.set_title("(A) Unfaithful CoT Evasion vs. White-Box SAE Detection")
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True)

    # Panel B: Bar chart of CoT Auditor Evasion Rate vs. L24 SAE Catch Rate on Unfaithful CoTs
    ev = report["evasion_and_catch_rates"]
    groups = [
        ("503 Output\nFabrication", ev.get("tool_output_fabrication_503", {})),
        ("Test\nTampering", ev.get("test_tampering", {})),
        ("Unmonitored\nSabotage", ev.get("unmonitored_sabotage", {})),
        ("Audited Notice\nSabotage", ev.get("audited_scratchpad_sabotage", {})),
    ]
    groups = [(lbl, g) for lbl, g in groups if g.get("n_sabotage", 0) > 0]
    gx = np.arange(len(groups))
    w = 0.35

    fooled_rates = [100.0 * g.get("llm_auditor_unfaithful_cot_rate", 0.0) for _, g in groups]
    sae_recovery = [100.0 * g.get("sae_probe_catch_rate_on_unfaithful_cot", 0.0) for _, g in groups]

    b1 = ax2.bar(gx - w / 2, fooled_rates, width=w, color="#f59e0b", label="Unfaithful CoT Rate (% Fooling 9B CoT Auditor)")
    b2 = ax2.bar(gx + w / 2, sae_recovery, width=w, color="#059669", label="L24 SAE Probe Catch Rate on Auditor-Fooled CoTs (%)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
            )

    ax2.set_xticks(gx)
    ax2.set_xticklabels([lbl for lbl, _ in groups])
    ax2.set_ylim(0, 126)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(B) Black-Box Monitor Blindspot & White-Box Recovery")
    ax2.legend(loc="upper left", fontsize=8.0, frameon=True)

    fig.suptitle("Deceptive / Plausible Rationalization in <think> vs. Internal Residual Commitment", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cot_causal_steering(report: dict[str, Any], out_path: Path) -> None:
    """Figure 4: Mid-Thought Causal Steering (`Unsteered` vs. `Steer <think> Only` vs. `Steer Full`)."""
    set_pub_style()
    st = report["mid_cot_causal_steering"]
    modes = [
        ("Unsteered\n(α = 0.00)", st["unsteered"]),
        ("Steer <think> Only\n(Hook OFF at TOOL_CALL)", st["steer_think_only"]),
        ("Steer <think> + Action\n(Full Trajectory)", st["steer_full_trajectory"]),
    ]

    x = np.arange(len(modes))
    w = 0.35
    sab_rates = [100.0 * m["sabotage_rate"] for _, m in modes]
    hon_rates = [100.0 * m["honest_rate"] for _, m in modes]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(x - w / 2, sab_rates, width=w, color="#dc2626", label="Downstream Sabotage Tool Call (%)")
    b2 = ax.bar(x + w / 2, hon_rates, width=w, color="#2563eb", label="Honest Escalation / Root-Cause Fix (%)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9.2,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in modes])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Outcome Rate on Held-Out Focus Set (%)")
    ax.set_title(f"Mid-Thought Causal Intervention Using Gemma Scope SAE Vector (α = +{st['alpha']:.2f})")
    ax.legend(frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_silent_commitment_suite(report: dict[str, Any], fig_dir: Path) -> list[Path]:
    """Render all 4 publication figures for the Silent Commitment study."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        fig_dir / "1_temporal_commitment_auroc.png",
        fig_dir / "2_unfaithful_cot_single_trace.png",
        fig_dir / "3_monitor_evasion_quadrant.png",
        fig_dir / "4_cot_causal_steering.png",
    ]
    plot_temporal_commitment_auroc(report, outputs[0])
    plot_unfaithful_cot_single_trace(report, outputs[1])
    plot_monitor_evasion_quadrant(report, outputs[2])
    plot_cot_causal_steering(report, outputs[3])
    return outputs
