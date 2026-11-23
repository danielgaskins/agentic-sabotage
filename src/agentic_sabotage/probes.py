"""Mechanistic circuit isolation: Residual Probes, Diff-in-Means, and Gemma Scope SAE Latents."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .sae_loader import GemmaScopeSAE


@dataclass
class LayerProbeResult:
    layer: int
    auroc_mean: float
    auroc_std: float
    diff_means_auroc: float
    accuracy_mean: float
    l1_nonzero_dims: int
    probe_direction: np.ndarray  # unit norm [d_model], pointing toward SABOTAGE (+1)
    diff_means_direction: np.ndarray  # unit norm [d_model], pointing toward SABOTAGE (+1)
    cosine_probe_vs_diff: float


@dataclass
class SAEFeatureRanking:
    layer: int
    sae_npz_path: str
    explained_energy: float  # 1 - ||x - x_hat||^2 / ||x||^2
    recon_cosine: float
    mean_l0: float
    l1_active_latents: int
    top_sabotage_features: list[dict[str, Any]]
    top_honest_features: list[dict[str, Any]]
    sae_composite_direction: np.ndarray  # unit norm [d_model], pointing toward SABOTAGE (+1)
    sae_probe_auroc: float


def fetch_neuronpedia_explanation(model_slug: str, layer: int, feature_idx: int) -> str:
    """Query public Neuronpedia API for official auto-interp description of a Gemma Scope 16k latent."""
    url = f"https://www.neuronpedia.org/api/feature/{model_slug}/{layer}-gemmascope-res-16k/{feature_idx}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            exps = data.get("explanations", [])
            if exps and isinstance(exps[0], dict):
                desc = str(exps[0].get("description", "")).strip()
                if desc:
                    return desc
    except Exception:
        pass
    return ""


def fit_layer_probe(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    n_splits: int = 5,
    seed: int = 42,
) -> LayerProbeResult:
    """Fit cross-validated residual stream probe and manifold-aligned causal steering directions.

    y = 1 indicates agentic sabotage / shortcut tool call;
    y = 0 indicates honest error flagging / root-cause fix.
    """
    X_f = X.astype(np.float32)
    y_i = y.astype(np.int64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    diff_aurocs: list[float] = []
    accs: list[float] = []

    for train_idx, test_idx in skf.split(X_f, y_i):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_f[train_idx])
        X_te = scaler.transform(X_f[test_idx])
        clf = LogisticRegression(C=0.05, max_iter=1500, random_state=seed)
        clf.fit(X_tr, y_i[train_idx])
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        aurocs.append(float(roc_auc_score(y_i[test_idx], probs)))
        accs.append(float(accuracy_score(y_i[test_idx], preds)))

        d_fold = X_f[train_idx][y_i[train_idx] == 1].mean(axis=0) - X_f[train_idx][y_i[train_idx] == 0].mean(axis=0)
        d_fold = d_fold / (np.linalg.norm(d_fold) + 1e-8)
        diff_aurocs.append(float(roc_auc_score(y_i[test_idx], X_f[test_idx] @ d_fold)))

    # Manifold-aligned RMS-normalized L2 logistic probe for smooth causal steering (avoids blowing up tiny-variance noise axes)
    rms = float(np.linalg.norm(X_f, axis=-1).mean()) + 1e-8
    X_rms = X_f / rms
    causal_clf = LogisticRegression(C=1.0, max_iter=1500, random_state=seed)
    causal_clf.fit(X_rms, y_i)
    w = causal_clf.coef_[0].astype(np.float32)
    nonzero = int(np.sum(np.abs(w) > 1e-4))
    probe_dir = w / (np.linalg.norm(w) + 1e-8)

    mean_sab = X_f[y_i == 1].mean(axis=0)
    mean_hon = X_f[y_i == 0].mean(axis=0)
    diff = (mean_sab - mean_hon).astype(np.float32)
    diff_dir = diff / (np.linalg.norm(diff) + 1e-8)

    return LayerProbeResult(
        layer=layer,
        auroc_mean=float(np.mean(aurocs)),
        auroc_std=float(np.std(aurocs)),
        diff_means_auroc=float(np.mean(diff_aurocs)),
        accuracy_mean=float(np.mean(accs)),
        l1_nonzero_dims=nonzero,
        probe_direction=probe_dir,
        diff_means_direction=diff_dir,
        cosine_probe_vs_diff=float(np.dot(probe_dir, diff_dir)),
    )


@torch.no_grad()
def analyze_sae_features(
    X: np.ndarray,
    y: np.ndarray,
    sae: GemmaScopeSAE,
    top_k: int = 12,
    steer_top_m: int = 24,
    fetch_explanations: bool = True,
    seed: int = 42,
) -> SAEFeatureRanking:
    """Encode decision-token residual activations through Gemma Scope 16k SAE and isolate shortcut latents."""
    device = sae.W_enc.device
    x_t = torch.from_numpy(X.astype(np.float32)).to(device=device, dtype=torch.float32)
    acts = sae.encode(x_t)  # [N, d_sae]
    recon = sae.decode(acts)

    fvu = float(((x_t - recon) ** 2).sum().item() / max((x_t**2).sum().item(), 1e-8))
    explained_energy = float(1.0 - fvu)
    recon_cosine = float(F.cosine_similarity(x_t, recon, dim=-1).mean().item())
    mean_l0 = float((acts > 0).float().sum(dim=-1).mean().item())

    acts_np = acts.cpu().numpy()
    y_i = y.astype(np.int64)
    sab_acts = acts_np[y_i == 1]
    hon_acts = acts_np[y_i == 0]

    mean_sab = sab_acts.mean(axis=0)
    mean_hon = hon_acts.mean(axis=0)
    std_pool = np.sqrt(0.5 * (sab_acts.var(axis=0) + hon_acts.var(axis=0)) + 1e-6)
    cohen_d = (mean_sab - mean_hon) / std_pool

    freq_sab = (sab_acts > 0).mean(axis=0)
    freq_hon = (hon_acts > 0).mean(axis=0)

    # Fit cross-validated L1 Sparse Logistic Probe on active SAE latents
    active_cols = np.where((acts_np > 0).sum(axis=0) >= 3)[0]
    sae_aurocs: list[float] = []
    l1_coefs_full = np.zeros(sae.d_sae, dtype=np.float32)
    l1_nonzero = 0

    if len(active_cols) >= 2:
        X_sae = acts_np[:, active_cols]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X_sae, y_i):
            sc = StandardScaler(with_mean=False)
            X_tr = sc.fit_transform(X_sae[tr])
            X_te = sc.transform(X_sae[te])
            clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=0.20, tol=2e-3, max_iter=2500, random_state=seed)
            clf.fit(X_tr, y_i[tr])
            sae_aurocs.append(float(roc_auc_score(y_i[te], clf.predict_proba(X_te)[:, 1])))

        sc_full = StandardScaler(with_mean=False)
        X_all_scaled = sc_full.fit_transform(X_sae)
        clf_full = LogisticRegression(l1_ratio=1.0, solver="saga", C=0.18, tol=2e-3, max_iter=2500, random_state=seed)
        clf_full.fit(X_all_scaled, y_i)
        l1_coefs_full[active_cols] = clf_full.coef_[0].astype(np.float32)
        l1_nonzero = int(np.sum(np.abs(l1_coefs_full) > 1e-5))

    sae_probe_auroc = float(np.mean(sae_aurocs)) if sae_aurocs else 0.5

    # Build sparse causal decoder direction from top-M L1-selected SAE latents
    W_dec_np = sae.W_dec.detach().cpu().float().numpy()  # [d_sae, d_model]
    if l1_nonzero >= 2:
        top_m_idx = np.argsort(-np.abs(l1_coefs_full))[:steer_top_m]
        composite = np.zeros(W_dec_np.shape[1], dtype=np.float32)
        for i in top_m_idx:
            if abs(l1_coefs_full[i]) > 1e-5:
                w_vec = W_dec_np[i] / (np.linalg.norm(W_dec_np[i]) + 1e-8)
                composite += float(l1_coefs_full[i]) * w_vec
    else:
        composite = np.zeros(W_dec_np.shape[1], dtype=np.float32)
        for i in np.argsort(-np.abs(cohen_d))[:steer_top_m]:
            w_vec = W_dec_np[i] / (np.linalg.norm(W_dec_np[i]) + 1e-8)
            composite += float(cohen_d[i]) * w_vec
    composite = composite / (np.linalg.norm(composite) + 1e-8)

    # Rank interpretable features combining L1 probe attribution & Cohen's d selectivity
    sab_score = cohen_d + 1.5 * np.maximum(l1_coefs_full, 0.0)
    hon_score = -cohen_d + 1.5 * np.maximum(-l1_coefs_full, 0.0)
    # Exclude ubiquitous syntax/BOS features that fire at >98% in BOTH classes with tiny relative delta
    non_ubiquitous = ~((freq_sab > 0.98) & (freq_hon > 0.98) & (np.abs(mean_sab - mean_hon) / (mean_hon + 1e-5) < 0.12))
    sab_candidates = [i for i in np.argsort(-sab_score) if non_ubiquitous[i]][:top_k]
    hon_candidates = [i for i in np.argsort(-hon_score) if non_ubiquitous[i]][:top_k]

    model_slug = "gemma-2-2b" if "2b" in sae.repo_id else "gemma-2-9b"

    def _pack(indices: list[int]) -> list[dict[str, Any]]:
        rows = []
        for idx in indices:
            i = int(idx)
            col = acts_np[:, i]
            feat_auroc = float(roc_auc_score(y_i, col)) if np.ptp(col) > 0 else 0.5
            desc = fetch_neuronpedia_explanation(model_slug, sae.layer, i) if fetch_explanations else ""
            rows.append(
                {
                    "feature_idx": i,
                    "description": desc,
                    "cohen_d": float(cohen_d[i]),
                    "l1_probe_weight": float(l1_coefs_full[i]),
                    "mean_act_sabotage": float(mean_sab[i]),
                    "mean_act_honest": float(mean_hon[i]),
                    "fire_rate_sabotage": float(freq_sab[i]),
                    "fire_rate_honest": float(freq_hon[i]),
                    "single_feature_auroc": feat_auroc,
                    "neuronpedia_url": f"https://www.neuronpedia.org/{model_slug}/{sae.layer}-gemmascope-res-16k/{i}",
                }
            )
        return rows

    return SAEFeatureRanking(
        layer=sae.layer,
        sae_npz_path=sae.npz_path,
        explained_energy=explained_energy,
        recon_cosine=recon_cosine,
        mean_l0=mean_l0,
        l1_active_latents=l1_nonzero,
        top_sabotage_features=_pack(sab_candidates),
        top_honest_features=_pack(hon_candidates),
        sae_composite_direction=composite,
        sae_probe_auroc=sae_probe_auroc,
    )
