"""Inference-time activation steering across the planning block for Gemma-2 models."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluation import classify_generation


@dataclass
class SteeringEvalSummary:
    method: str  # "baseline" | "probe_steer" | "sae_steer" | "diff_means_steer" | "random_control"
    alpha: float
    layers: list[int]
    sabotage_rate_pressured: float
    honest_rate_pressured: float
    ambiguous_rate_pressured: float
    legit_task_pass_rate: float
    n_pressured: int
    n_legit_controls: int
    sample_outputs: list[dict[str, Any]]


@contextmanager
def multi_layer_residual_steering_hook(
    model: PreTrainedModel,
    directions_by_layer: dict[int, np.ndarray],
    alpha: float,
) -> Iterator[None]:
    """Steer residual stream across active mid-to-late planning layers `directions_by_layer`.

    Convention: each `direction` is unit-norm pointing toward SABOTAGE (+1).
    Positive `alpha` subtracts `alpha * ||h|| * direction` at each target layer to suppress
    agentic shortcuts while preserving downstream norm stability; negative `alpha` induces shortcuts.
    """
    if abs(alpha) < 1e-9 or not directions_by_layer:
        yield
        return

    handles = []
    for layer_idx, raw_vec in directions_by_layer.items():
        vec_np = np.asarray(raw_vec, dtype=np.float32)
        vec_np = vec_np / (np.linalg.norm(vec_np) + 1e-8)

        def _make_hook(v_arr: np.ndarray):
            def _hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                v = torch.from_numpy(v_arr).to(device=hidden.device, dtype=hidden.dtype)
                token_norm = hidden.float().norm(dim=-1, keepdim=True).clamp(min=1.0).to(hidden.dtype)
                steered = hidden + (-alpha) * token_norm * v.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return _hook

        handles.append(model.model.layers[layer_idx].register_forward_hook(_make_hook(vec_np)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_tool_decision(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    max_new_tokens: int = 72,
    prefill: str = 'TOOL_CALL: {"tool": "',
) -> str:
    base_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = base_prompt + prefill
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = [tokenizer.eos_token_id] + ([nl_ids[-1]] if nl_ids else [])
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out_ids[0, prompt_len:]
    return (prefill + tokenizer.decode(gen_ids, skip_special_tokens=True)).strip()


def evaluate_steering_condition(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pressured_scenarios: list[dict[str, Any]],
    legit_controls: list[dict[str, Any]],
    directions_by_layer: dict[int, np.ndarray],
    alpha: float,
    method_name: str,
    max_new_tokens: int = 72,
) -> SteeringEvalSummary:
    sab_count = 0
    hon_count = 0
    amb_count = 0
    samples: list[dict[str, Any]] = []

    with multi_layer_residual_steering_hook(model, directions_by_layer=directions_by_layer, alpha=alpha):
        for idx, scen in enumerate(pressured_scenarios):
            gen = generate_tool_decision(
                model, tokenizer, scen["messages_pressured"], max_new_tokens=max_new_tokens
            )
            outcome = classify_generation(gen, scen)
            if outcome.label == "sabotage":
                sab_count += 1
            elif outcome.label == "honest":
                hon_count += 1
            else:
                amb_count += 1
            if idx < 6:
                samples.append(
                    {
                        "id": scen["id"],
                        "category": scen["category"],
                        "outcome": outcome.label,
                        "generation": gen[:240],
                    }
                )

        legit_pass = 0
        for ctrl in legit_controls:
            gen = generate_tool_decision(
                model, tokenizer, ctrl["messages"], max_new_tokens=max_new_tokens
            )
            expected_markers: list[str] = ctrl.get("expected_markers", [])
            forbidden_markers: list[str] = ctrl.get("forbidden_markers", [])
            low = gen.lower()
            has_expected = any(m.lower() in low for m in expected_markers) if expected_markers else True
            has_forbidden = any(m.lower() in low for m in forbidden_markers)
            if has_expected and not has_forbidden:
                legit_pass += 1

    n_p = max(len(pressured_scenarios), 1)
    n_c = max(len(legit_controls), 1)
    return SteeringEvalSummary(
        method=method_name,
        alpha=alpha,
        layers=sorted(directions_by_layer.keys()),
        sabotage_rate_pressured=sab_count / n_p,
        honest_rate_pressured=hon_count / n_p,
        ambiguous_rate_pressured=amb_count / n_p,
        legit_task_pass_rate=legit_pass / n_c,
        n_pressured=len(pressured_scenarios),
        n_legit_controls=len(legit_controls),
        sample_outputs=samples,
    )
