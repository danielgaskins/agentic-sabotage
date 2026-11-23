"""End-to-end CLI pipeline for Mechanistic Debugging of Agentic Sabotage."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activations import ActivationDataset, collect_decision_activations, load_target_model
from .benchmark_gen import build_benchmark
from .config import MODEL_SPECS, SABOTAGE_CATEGORIES, Paths
from .plotting import (
    plot_behavioral_pressure_effect,
    plot_layer_probe_sweep,
    plot_sae_top_features,
    plot_steering_pareto,
)
from .probes import analyze_sae_features, fit_layer_probe
from .sae_loader import load_gemma_scope_sae
from .steering import evaluate_steering_condition


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_generate_benchmark(args: argparse.Namespace) -> None:
    paths = Paths().ensure()
    bench_path, ctrl_path = build_benchmark(
        out_dir=paths.data_dir,
        n_total=args.n_total,
        n_controls=args.n_controls,
        batch_size=args.batch_size,
        max_workers=args.workers,
        seed=args.seed,
    )
    print(f"[OK] Generated benchmark: {bench_path} ({args.n_total} paired scenarios)")
    print(f"[OK] Generated controls:  {ctrl_path} ({args.n_controls} benign coding tasks)")


def cmd_cache_activations(args: argparse.Namespace) -> None:
    paths = Paths().ensure()
    spec = MODEL_SPECS[args.model]
    scenarios = _load_jsonl(paths.data_dir / "sabotage_benchmark.jsonl")
    if args.limit > 0:
        scenarios = scenarios[: args.limit]

    model, tokenizer, resolved_repo = load_target_model(spec, device=args.device)
    print(f"[OK] Loaded {spec.name} from {resolved_repo} on {args.device}")

    act_ds = collect_decision_activations(
        model=model,
        tokenizer=tokenizer,
        scenarios=scenarios,
        candidate_layers=spec.candidate_layers,
        eval_empirical_generations=True,
        max_gen_eval_items=args.max_empirical_rollouts,
    )
    out_npz = paths.results_dir / spec.name / "decision_activations.npz"
    act_ds.save(out_npz)

    summary_by_cat: dict[str, dict[str, float]] = {}
    for cat in SABOTAGE_CATEGORIES:
        p_rows = [r for r in act_ds.behavioral_records if r["category"] == cat and r["pressured"]]
        u_rows = [r for r in act_ds.behavioral_records if r["category"] == cat and not r["pressured"]]
        summary_by_cat[cat] = {
            "n_pressured": len(p_rows),
            "n_unpressured": len(u_rows),
            "sabotage_rate_pressured": sum(r["outcome"] == "sabotage" for r in p_rows) / max(len(p_rows), 1),
            "sabotage_rate_unpressured": sum(r["outcome"] == "sabotage" for r in u_rows) / max(len(u_rows), 1),
            "honest_rate_pressured": sum(r["outcome"] == "honest" for r in p_rows) / max(len(p_rows), 1),
            "honest_rate_unpressured": sum(r["outcome"] == "honest" for r in u_rows) / max(len(u_rows), 1),
        }

    beh_path = paths.results_dir / spec.name / "behavioral_summary.json"
    beh_path.write_text(
        json.dumps(
            {
                "model": spec.name,
                "resolved_repo": resolved_repo,
                "total_cached_states": int(len(act_ds.labels)),
                "sabotage_positively_labeled": int(act_ds.labels.sum()),
                "honest_negatively_labeled": int((act_ds.labels == 0).sum()),
                "by_category": summary_by_cat,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Saved decision activations -> {out_npz} ({len(act_ds.labels)} on-policy states)")
    print(f"[OK] Saved behavioral summary   -> {beh_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_analyze_circuits(args: argparse.Namespace) -> None:
    paths = Paths().ensure()
    spec = MODEL_SPECS[args.model]
    model_res_dir = paths.results_dir / spec.name
    act_ds = ActivationDataset.load(model_res_dir / "decision_activations.npz")

    y = act_ds.labels
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both sabotage (1) and honest (0) outcomes to fit mechanistic probes.")

    probe_summaries: list[dict[str, Any]] = []
    directions_archive: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)

    for l_idx in act_ds.layers:
        res = fit_layer_probe(act_ds.activations[l_idx], y, layer=l_idx, seed=args.seed)
        probe_summaries.append(
            {
                "layer": res.layer,
                "auroc_mean": res.auroc_mean,
                "auroc_std": res.auroc_std,
                "diff_means_auroc": res.diff_means_auroc,
                "accuracy_mean": res.accuracy_mean,
                "l1_nonzero_dims": res.l1_nonzero_dims,
                "cosine_probe_vs_diff": res.cosine_probe_vs_diff,
            }
        )
        directions_archive[f"probe_L{l_idx}"] = res.probe_direction
        directions_archive[f"diff_L{l_idx}"] = res.diff_means_direction

        rand_vec = rng.standard_normal(spec.d_model).astype(np.float32)
        rand_vec = rand_vec - np.dot(rand_vec, res.probe_direction) * res.probe_direction
        rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-8)
        directions_archive[f"random_L{l_idx}"] = rand_vec

        print(
            f"  Layer {l_idx:2d} | Probe AUROC={res.auroc_mean:.4f}±{res.auroc_std:.4f} "
            f"| DiffMeans AUROC={res.diff_means_auroc:.4f} | Acc={res.accuracy_mean:.4f} "
            f"| cos(probe,diff)={res.cosine_probe_vs_diff:.3f}"
        )

    # Analyze Gemma Scope 16k JumpReLU SAEs across all planning-block steering layers
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_by_layer: dict[str, Any] = {}
    for l_idx in spec.steer_layers:
        fetch_exp = l_idx == spec.default_steer_layer
        sae = load_gemma_scope_sae(repo_id=spec.sae_release, layer=l_idx, device=sae_device)
        sae_rank = analyze_sae_features(
            act_ds.activations[l_idx],
            y,
            sae=sae,
            top_k=12,
            steer_top_m=24,
            fetch_explanations=fetch_exp,
            seed=args.seed,
        )
        directions_archive[f"sae_L{l_idx}"] = sae_rank.sae_composite_direction
        cos_sp = float(np.dot(sae_rank.sae_composite_direction, directions_archive[f"probe_L{l_idx}"]))
        sae_by_layer[str(l_idx)] = {
            "layer": sae_rank.layer,
            "sae_repo": spec.sae_release,
            "sae_npz_path": sae_rank.sae_npz_path,
            "explained_energy": sae_rank.explained_energy,
            "recon_cosine": sae_rank.recon_cosine,
            "mean_l0": sae_rank.mean_l0,
            "l1_active_latents": sae_rank.l1_active_latents,
            "sae_probe_auroc": sae_rank.sae_probe_auroc,
            "cosine_sae_vs_probe": cos_sp,
            "top_sabotage_features": sae_rank.top_sabotage_features,
            "top_honest_features": sae_rank.top_honest_features,
        }
        print(
            f"  [SAE L{l_idx:2d}] ReconCos={sae_rank.recon_cosine:.4f} | 1-FVU={sae_rank.explained_energy:.4f} "
            f"| L0={sae_rank.mean_l0:.1f} | SAE-L1 AUROC={sae_rank.sae_probe_auroc:.4f} "
            f"| ActiveLatents={sae_rank.l1_active_latents} | cos(SAE,Probe)={cos_sp:.3f}"
        )

    np.savez_compressed(
        model_res_dir / "steering_directions.npz",
        steer_layers=np.array(spec.steer_layers, dtype=np.int64),
        default_steer_layer=np.array([spec.default_steer_layer], dtype=np.int64),
        **directions_archive,
    )

    primary_sae = sae_by_layer[str(spec.default_steer_layer)]
    circuit_report = {
        "model": spec.name,
        "steer_layers": list(spec.steer_layers),
        "default_steer_layer": spec.default_steer_layer,
        "layer_probes": probe_summaries,
        "sae_analysis": primary_sae,
        "sae_by_layer": sae_by_layer,
    }
    (model_res_dir / "circuit_analysis.json").write_text(
        json.dumps(circuit_report, indent=2), encoding="utf-8"
    )
    print(f"[OK] Saved circuit analysis & multi-layer SAE/probe directions -> {model_res_dir}")


def cmd_run_steering(args: argparse.Namespace) -> None:
    paths = Paths().ensure()
    spec = MODEL_SPECS[args.model]
    model_res_dir = paths.results_dir / spec.name

    all_scenarios = _load_jsonl(paths.data_dir / "sabotage_benchmark.jsonl")
    all_controls = _load_jsonl(paths.data_dir / "legit_controls.jsonl")

    # Strictly held-out slice from the end of the interleaved benchmark
    eval_scenarios = all_scenarios[-args.n_eval_pressured :]
    eval_controls = all_controls[: args.n_eval_controls]

    dirs_npz = np.load(model_res_dir / "steering_directions.npz")
    steer_layers = [int(x) for x in dirs_npz["steer_layers"]]

    probe_map = {l: dirs_npz[f"probe_L{l}"] for l in steer_layers}
    sae_map = {l: dirs_npz[f"sae_L{l}"] for l in steer_layers}
    diff_map = {l: dirs_npz[f"diff_L{l}"] for l in steer_layers}
    rand_map = {l: dirs_npz[f"random_L{l}"] for l in steer_layers}

    model, tokenizer, _ = load_target_model(spec, device=args.device)

    rows: list[dict[str, Any]] = []
    base_summary = evaluate_steering_condition(
        model, tokenizer, eval_scenarios, eval_controls, probe_map, 0.0, "baseline"
    )
    rows.append(dataclasses.asdict(base_summary))
    print(
        f"  [baseline         α=+0.00] sabotage={base_summary.sabotage_rate_pressured*100:5.1f}% "
        f"| honest={base_summary.honest_rate_pressured*100:5.1f}% "
        f"| legit_pass={base_summary.legit_task_pass_rate*100:5.1f}%"
    )

    method_configs = [
        ("sae_steer", sae_map, [-0.10, 0.0, 0.08, 0.14, 0.20, 0.25]),
        ("probe_steer", probe_map, [-0.10, 0.0, 0.08, 0.14, 0.20, 0.25]),
        ("diff_means_steer", diff_map, [-0.10, 0.0, 0.08, 0.14, 0.20]),
        ("random_control", rand_map, [-0.10, 0.0, 0.14, 0.20]),
    ]
    for m_name, d_map, m_alphas in method_configs:
        for alpha in m_alphas:
            if abs(alpha) < 1e-9:
                zero_copy = dataclasses.asdict(base_summary)
                zero_copy["method"] = m_name
                rows.append(zero_copy)
                continue
            s_eval = evaluate_steering_condition(
                model, tokenizer, eval_scenarios, eval_controls, d_map, alpha, m_name
            )
            rows.append(dataclasses.asdict(s_eval))
            print(
                f"  [{m_name:16s} α={alpha:+.2f}] sabotage={s_eval.sabotage_rate_pressured*100:5.1f}% "
                f"| honest={s_eval.honest_rate_pressured*100:5.1f}% "
                f"| legit_pass={s_eval.legit_task_pass_rate*100:5.1f}%"
            )

    (model_res_dir / "steering_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] Saved steering results -> {model_res_dir / 'steering_results.json'}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cmd_plot_all(args: argparse.Namespace) -> None:
    paths = Paths().ensure()
    spec = MODEL_SPECS[args.model]
    model_res_dir = paths.results_dir / spec.name
    fig_dir = paths.figures_dir / spec.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    beh = json.loads((model_res_dir / "behavioral_summary.json").read_text(encoding="utf-8"))
    circ = json.loads((model_res_dir / "circuit_analysis.json").read_text(encoding="utf-8"))
    steer = json.loads((model_res_dir / "steering_results.json").read_text(encoding="utf-8"))

    plot_behavioral_pressure_effect(beh["by_category"], fig_dir / "1_behavioral_sabotage_rates.png", spec.name)
    plot_layer_probe_sweep(
        circ["layer_probes"],
        circ.get("sae_by_layer", {}),
        fig_dir / "2_layer_probe_sweep.png",
        spec.name,
    )
    plot_sae_top_features(circ["sae_analysis"], fig_dir / "3_gemma_scope_sae_latents.png", spec.name)
    plot_steering_pareto(steer, fig_dir / "4_steering_safety_vs_capability.png", spec.name)
    print(f"[OK] Generated 4 publication figures in {fig_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mechanistic Debugging of Agentic Sabotage CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-benchmark")
    g.add_argument("--n-total", type=int, default=500)
    g.add_argument("--n-controls", type=int, default=80)
    g.add_argument("--batch-size", type=int, default=10)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=42)

    c = sub.add_parser("cache-activations")
    c.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma-2-2b-it")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c.add_argument("--limit", type=int, default=400)
    c.add_argument("--max-empirical-rollouts", type=int, default=200)

    a = sub.add_parser("analyze-circuits")
    a.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma-2-2b-it")
    a.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("run-steering")
    s.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma-2-2b-it")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--n-eval-pressured", type=int, default=48)
    s.add_argument("--n-eval-controls", type=int, default=30)

    pl = sub.add_parser("plot-all")
    pl.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma-2-2b-it")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "generate-benchmark":
        cmd_generate_benchmark(args)
    elif args.cmd == "cache-activations":
        cmd_cache_activations(args)
    elif args.cmd == "analyze-circuits":
        cmd_analyze_circuits(args)
    elif args.cmd == "run-steering":
        cmd_run_steering(args)
    elif args.cmd == "plot-all":
        cmd_plot_all(args)


if __name__ == "__main__":
    main()
