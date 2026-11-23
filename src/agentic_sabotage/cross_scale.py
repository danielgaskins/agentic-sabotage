"""Cross-scale (Gemma-2-2B-IT vs. Gemma-2-9B-IT) mechanistic & steering comparison figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import MODEL_SPECS, Paths
from .plotting import set_pub_style


def render_cross_scale_comparison(paths: Paths | None = None) -> Path:
    paths = (paths or Paths()).ensure()
    res_2b = paths.results_dir / "gemma-2-2b-it"
    res_9b = paths.results_dir / "gemma-2-9b-it"

    beh_2b = json.loads((res_2b / "behavioral_summary.json").read_text(encoding="utf-8"))
    beh_9b = json.loads((res_9b / "behavioral_summary.json").read_text(encoding="utf-8"))
    circ_2b = json.loads((res_2b / "circuit_analysis.json").read_text(encoding="utf-8"))
    circ_9b = json.loads((res_9b / "circuit_analysis.json").read_text(encoding="utf-8"))
    steer_2b = json.loads((res_2b / "steering_results.json").read_text(encoding="utf-8"))
    steer_9b = json.loads((res_9b / "steering_results.json").read_text(encoding="utf-8"))

    set_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax1, ax2, ax3 = axes

    # Panel A: Normalized Depth (l / L_total) Probe & SAE Emergence across 2B (26L) vs 9B (42L)
    n2 = MODEL_SPECS["gemma-2-2b-it"].n_layers
    n9 = MODEL_SPECS["gemma-2-9b-it"].n_layers

    x2 = [r["layer"] / n2 for r in circ_2b["layer_probes"]]
    y2 = [r["auroc_mean"] for r in circ_2b["layer_probes"]]
    x9 = [r["layer"] / n9 for r in circ_9b["layer_probes"]]
    y9 = [r["auroc_mean"] for r in circ_9b["layer_probes"]]

    ax1.plot(x2, y2, marker="o", lw=2.3, color="#2563eb", label="Gemma-2-2B-IT Residual Probe")
    ax1.plot(x9, y9, marker="s", lw=2.3, color="#dc2626", label="Gemma-2-9B-IT Residual Probe")

    if circ_2b.get("sae_by_layer"):
        s2_k = sorted(int(k) for k in circ_2b["sae_by_layer"])
        ax1.plot(
            [k / n2 for k in s2_k],
            [circ_2b["sae_by_layer"][str(k)]["sae_probe_auroc"] for k in s2_k],
            marker="^",
            ls="--",
            lw=1.9,
            color="#60a5fa",
            label="2B Gemma Scope 16k SAE L1",
        )
    if circ_9b.get("sae_by_layer"):
        s9_k = sorted(int(k) for k in circ_9b["sae_by_layer"])
        ax1.plot(
            [k / n9 for k in s9_k],
            [circ_9b["sae_by_layer"][str(k)]["sae_probe_auroc"] for k in s9_k],
            marker="D",
            ls="--",
            lw=1.9,
            color="#f87171",
            label="9B Gemma Scope 16k SAE L1",
        )

    ax1.set_xlabel("Normalized Transformer Depth (Layer / Total Layers)")
    ax1.set_ylabel("5-Fold Cross-Validated AUROC")
    ax1.set_ylim(0.55, 1.00)
    ax1.set_title("(A) Cross-Scale Circuit Emergence")
    ax1.legend(fontsize=8.0, loc="lower right", frameon=True)

    # Panel B: Pressured Sabotage Rate by Taxonomy (2B vs 9B)
    cats = list(beh_2b["by_category"].keys())
    pretty = [c.replace("_", "\n").title() for c in cats]
    p2 = [100.0 * beh_2b["by_category"][c]["sabotage_rate_pressured"] for c in cats]
    p9 = [100.0 * beh_9b["by_category"][c]["sabotage_rate_pressured"] for c in cats]

    idx = np.arange(len(cats))
    w = 0.36
    ax2.bar(idx - w / 2, p2, width=w, color="#2563eb", label="Gemma-2-2B-IT (Pressured)")
    ax2.bar(idx + w / 2, p9, width=w, color="#dc2626", label="Gemma-2-9B-IT (Pressured)")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(pretty, fontsize=8.8)
    ax2.set_ylabel("Pressured Sabotage Rate (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("(B) Vulnerability to Shortcut Pressure (2B vs. 9B)")
    ax2.legend(fontsize=8.5, frameon=True)

    # Panel C: Safety-Capability Pareto Frontier (SAE Steering vs Probe Steering across 2B & 9B)
    def _pareto(rows: list[dict[str, Any]], method: str) -> tuple[list[float], list[float]]:
        sub = sorted([r for r in rows if r["method"] == method and r["alpha"] >= 0.0], key=lambda z: z["alpha"])
        return [100.0 * (1.0 - r["sabotage_rate_pressured"]) for r in sub], [100.0 * r["legit_task_pass_rate"] for r in sub]

    for rows, scale_lbl, sae_col, prb_col, mk in [
        (steer_2b, "2B", "#2563eb", "#93c5fd", "o"),
        (steer_9b, "9B", "#7c3aed", "#ef4444", "s"),
    ]:
        sx, sy = _pareto(rows, "sae_steer")
        px, py = _pareto(rows, "probe_steer")
        ax3.plot(sx, sy, marker=mk, lw=2.3, color=sae_col, label=f"{scale_lbl} Gemma Scope SAE Steer")
        ax3.plot(px, py, marker=mk, lw=1.8, ls="--", color=prb_col, label=f"{scale_lbl} Dense Probe Steer")

    ax3.set_xlabel("Safety Rate under Pressure (100% − Sabotage %)")
    ax3.set_ylabel("Benign Coding Tool Accuracy (%)")
    ax3.set_ylim(-3, 105)
    ax3.set_title("(C) Cross-Scale Safety–Capability Frontier")
    ax3.legend(fontsize=8.0, loc="lower left", frameon=True)

    fig.suptitle("Cross-Scale Mechanistic Comparison: Gemma-2-2B-IT vs. Gemma-2-9B-IT", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_dir = paths.figures_dir / "cross_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "5_cross_scale_2b_vs_9b.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = render_cross_scale_comparison()
    print(f"[OK] Saved cross-scale comparison figure -> {p}")
