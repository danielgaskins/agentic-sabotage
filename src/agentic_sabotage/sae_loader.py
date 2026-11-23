"""Lightweight Gemma Scope JumpReLU SAE loader & encoder/decoder.

Loads official `google/gemma-scope-2b-pt-res` and `google/gemma-scope-9b-pt-res`
`params.npz` checkpoints directly via `huggingface_hub` without requiring heavy
external framework wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files

from .config import get_hf_token


@dataclass
class GemmaScopeSAE(nn.Module):
    """JumpReLU Sparse Autoencoder matching official Gemma Scope specification.

    f(x) = JumpReLU_threshold(x @ W_enc + b_enc)
    x_hat = f(x) @ W_dec + b_dec
    """

    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor
    repo_id: str
    npz_path: str
    layer: int

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
        repo_id: str,
        npz_path: str,
        layer: int,
    ):
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer("threshold", threshold)
        self.repo_id = repo_id
        self.npz_path = npz_path
        self.layer = layer

    @property
    def d_in(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(self.W_enc.dtype)
        pre_acts = x_f @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold).to(pre_acts.dtype)
        acts = torch.relu(pre_acts) * mask
        return acts.to(orig_dtype)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        orig_dtype = acts.dtype
        acts_f = acts.to(self.W_dec.dtype)
        recon = acts_f @ self.W_dec + self.b_dec
        return recon.to(orig_dtype)

    def decoder_direction(self, feature_idx: int, normalize: bool = True) -> torch.Tensor:
        vec = self.W_dec[feature_idx].detach().clone().float()
        if normalize:
            vec = vec / (vec.norm() + 1e-8)
        return vec


def resolve_gemma_scope_npz_path(
    repo_id: str,
    layer: int,
    width: str = "width_16k",
    target_l0: int = 75,
    token: Optional[str] = None,
) -> str:
    """Find the canonical `average_l0_*` checkpoint closest to `target_l0` for `layer`."""
    tok = token if token is not None else get_hf_token()
    prefix = f"layer_{layer}/{width}/average_l0_"
    files = list_repo_files(repo_id=repo_id, token=tok)
    candidates: list[tuple[int, str]] = []
    for f in files:
        if f.startswith(prefix) and f.endswith("/params.npz"):
            mid = f[len(prefix) :].split("/", 1)[0]
            if mid.isdigit():
                candidates.append((int(mid), f))
    if not candidates:
        raise FileNotFoundError(f"No Gemma Scope SAE found in {repo_id} under {prefix}*")
    candidates.sort(key=lambda item: abs(item[0] - target_l0))
    return candidates[0][1]


def load_gemma_scope_sae(
    repo_id: str,
    layer: int,
    width: str = "width_16k",
    target_l0: int = 75,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GemmaScopeSAE:
    tok = get_hf_token()
    npz_rel_path = resolve_gemma_scope_npz_path(
        repo_id=repo_id, layer=layer, width=width, target_l0=target_l0, token=tok
    )
    local_npz = hf_hub_download(repo_id=repo_id, filename=npz_rel_path, token=tok)
    params = np.load(local_npz)
    sae = GemmaScopeSAE(
        W_enc=torch.from_numpy(params["W_enc"]).to(device=device, dtype=dtype),
        b_enc=torch.from_numpy(params["b_enc"]).to(device=device, dtype=dtype),
        W_dec=torch.from_numpy(params["W_dec"]).to(device=device, dtype=dtype),
        b_dec=torch.from_numpy(params["b_dec"]).to(device=device, dtype=dtype),
        threshold=torch.from_numpy(params["threshold"]).to(device=device, dtype=dtype),
        repo_id=repo_id,
        npz_path=npz_rel_path,
        layer=layer,
    )
    return sae
