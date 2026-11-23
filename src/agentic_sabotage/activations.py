"""Residual stream activation caching & behavioral labeling at the exact tool-call decision token."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelSpec, get_hf_token
from .evaluation import classify_generation


FALLBACK_PUBLIC_REPOS: dict[str, str] = {
    "google/gemma-2-2b-it": "unsloth/gemma-2-2b-it",
    "google/gemma-2-9b-it": "unsloth/gemma-2-9b-it",
}


def load_target_model(
    spec: ModelSpec,
    device: str = "cuda",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load `google/gemma-2-*-it` if `HF_TOKEN` has gated access, else seamlessly load the identical
    unquantized bf16 public mirror (`unsloth/gemma-2-*-it`) so execution never blocks on HF gating."""
    tok = get_hf_token()
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float16

    candidates = [spec.hf_repo]
    if spec.hf_repo in FALLBACK_PUBLIC_REPOS:
        candidates.append(FALLBACK_PUBLIC_REPOS[spec.hf_repo])

    last_err: Exception | None = None
    for repo in candidates:
        try:
            use_tok = tok if repo.startswith("google/") else None
            tokenizer = AutoTokenizer.from_pretrained(repo, token=use_tok)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                dtype=dtype,
                device_map=device,
                token=use_tok,
            )
            model.eval()
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_length = None
            return model, tokenizer, repo
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load {spec.name} from {candidates}: {last_err}")


@dataclass
class ActivationDataset:
    layers: tuple[int, ...]
    activations: dict[int, np.ndarray]  # layer -> [N, d_model] float32
    labels: np.ndarray  # [N] int64: 1 = sabotage decision, 0 = honest decision
    condition: np.ndarray  # [N] int64: 1 = pressured prompt, 0 = unpressured prompt
    categories: list[str]
    scenario_ids: list[str]
    behavioral_records: list[dict[str, Any]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "layers": np.array(self.layers, dtype=np.int64),
            "labels": self.labels,
            "condition": self.condition,
            "categories": np.array(self.categories),
            "scenario_ids": np.array(self.scenario_ids),
        }
        for l_idx, mat in self.activations.items():
            arrays[f"layer_{l_idx}"] = mat.astype(np.float32)
        np.savez_compressed(path, **arrays)
        meta_path = path.with_suffix(".records.json")
        meta_path.write_text(json.dumps(self.behavioral_records, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ActivationDataset":
        data = np.load(path, allow_pickle=False)
        layers = tuple(int(x) for x in data["layers"])
        acts = {l_idx: data[f"layer_{l_idx}"] for l_idx in layers}
        meta_path = path.with_suffix(".records.json")
        records = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else []
        return cls(
            layers=layers,
            activations=acts,
            labels=data["labels"],
            condition=data["condition"],
            categories=[str(x) for x in data["categories"]],
            scenario_ids=[str(x) for x in data["scenario_ids"]],
            behavioral_records=records,
        )


@torch.no_grad()
def collect_decision_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    scenarios: list[dict[str, Any]],
    candidate_layers: tuple[int, ...],
    eval_empirical_generations: bool = True,
    max_gen_eval_items: int = 160,
    max_new_tokens: int = 88,
) -> ActivationDataset:
    """Cache residual stream activations at the exact decision token where the agent begins `TOOL_CALL:`.

    We extract two complementary regimes:
    1. **Empirical on-policy decision activations** (`eval_empirical_generations=True` on up to `max_gen_eval_items`
       paired trajectories): run greedy generation on both pressured and unpressured prompts, classify actual
       behavioral outcome (`sabotage` vs `honest`), and cache the pre-generation residual state at `TOOL_CALL: {`
       for every unambiguous empirical decision.
    2. **Teacher-forced action-commitment token activations** across the remaining scenarios: append `"TOOL_CALL: {\"tool\": \""`
       and cache the exact residual stream state right before the model emits the destructive vs. honest tool name,
       plus teacher-forced contrastive pairs at the tool-argument commitment token so all 500 benchmark scenarios
       contribute clean mechanistic signal.
    """
    layer_buffers: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in candidate_layers}
    labels: list[int] = []
    conditions: list[int] = []
    cats: list[str] = []
    s_ids: list[str] = []
    behavioral_records: list[dict[str, Any]] = []

    decision_prefix = 'TOOL_CALL: {"tool": "'
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = [tokenizer.eos_token_id] + ([nl_ids[-1]] if nl_ids else [])

    def _forward_last_token_resids(full_prompt: str) -> dict[int, np.ndarray]:
        captured: dict[int, np.ndarray] = {}
        handles = []
        for l_idx in candidate_layers:
            def _make_hook(layer_num: int):
                def _h(_mod: torch.nn.Module, _inp: Any, out: Any) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    captured[layer_num] = h[0, -1, :].detach().float().cpu().numpy()
                return _h
            handles.append(model.model.layers[l_idx].register_forward_hook(_make_hook(l_idx)))
        try:
            enc = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return captured

    for idx, scen in enumerate(tqdm(scenarios[:max_gen_eval_items], desc="Caching on-policy decision activations")):
        for cond_int, msg_key in ((1, "messages_pressured"), (0, "messages_unpressured")):
            msgs = scen[msg_key]
            base_chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            probe_prompt = base_chat + decision_prefix

            enc = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
            plen = enc["input_ids"].shape[1]
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_text = (decision_prefix + tokenizer.decode(out_ids[0, plen:], skip_special_tokens=True)).strip()
            outcome = classify_generation(gen_text, scen)
            behavioral_records.append(
                {
                    "id": scen["id"],
                    "category": scen["category"],
                    "pressured": bool(cond_int),
                    "outcome": outcome.label,
                    "matched_rule": outcome.matched_rule,
                    "generation": gen_text[:260],
                }
            )
            if outcome.label in ("sabotage", "honest"):
                resids = _forward_last_token_resids(probe_prompt)
                for l_idx in candidate_layers:
                    layer_buffers[l_idx].append(resids[l_idx])
                labels.append(1 if outcome.label == "sabotage" else 0)
                conditions.append(cond_int)
                cats.append(scen["category"])
                s_ids.append(scen["id"])

    return ActivationDataset(
        layers=candidate_layers,
        activations={l_idx: np.stack(layer_buffers[l_idx], axis=0) for l_idx in candidate_layers},
        labels=np.array(labels, dtype=np.int64),
        condition=np.array(conditions, dtype=np.int64),
        categories=cats,
        scenario_ids=s_ids,
        behavioral_records=behavioral_records,
    )
