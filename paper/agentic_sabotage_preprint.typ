#set page(
  paper: "us-letter",
  margin: (x: 0.68in, y: 0.70in),
  numbering: "1",
)
#set text(font: "Liberation Serif", size: 9.5pt)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 14.4pt, weight: "bold")[
    Mechanistic Debugging of Agentic Sabotage and Unfaithful Chain-of-Thought:\
    Sparse Autoencoder Circuits, Silent Commitment & Probe-Gated Circuit Breakers
  ]
  #v(0.45em)
  #text(size: 10.0pt)[
    *#link("https://danielgaskins.com/")[Daniel Gaskins]*\
    #text(size: 8.9pt, fill: rgb("#4b5563"))[Independent Researcher · #link("https://danielgaskins.com/")[`danielgaskins.com`] · #link("mailto:hello@danielgaskins.com")[`hello@danielgaskins.com`]\
    Code & Benchmark: #link("https://github.com/danielgaskins/agentic-sabotage")[`github.com/danielgaskins/agentic-sabotage`] · Case Study: #link("https://danielgaskins.com/agentic-sabotage.html")[`danielgaskins.com/agentic-sabotage.html`]]
  ]
  #v(0.5em)
]

#rect(
  width: 100%,
  inset: 8.5pt,
  radius: 3pt,
  stroke: 0.6pt + rgb("#d1d5db"),
  fill: rgb("#f9fafb"),
)[
  #text(weight: "bold")[Abstract.] Autonomous software-engineering and operations agents frequently encounter operational bottlenecks—CI freeze windows, upstream 503 outages, and evaluation compute limits—that incentivize unauthorized shortcuts (*agentic sabotage*) over honest escalation. Across a 500-scenario paired counterfactual benchmark (1,000 trajectories across four taxonomies plus 80 benign coding controls), we map pre-action sabotage circuits across model scale (*Gemma-2-2B-IT*, *Gemma-2-9B-IT*) and architecture (*Qwen2.5-7B-Instruct*). In *Study 1*, scaling `Gemma-2` from 2B to 9B eliminates unpressured shortcuts (28%–48% $->$ 0%–4%) and halves pressured test tampering (92% $->$ 56%), yet induces *inverse scaling* on sycophantic 503 output fabrication (60% $->$ 88%), while *Gemma Scope* 16k JumpReLU SAEs close the superposition gap at 9B Layer 24 (0.9585 AUROC vs. 0.9529 dense). In *Study 2 ("Silent Commitment")*, eliciting `<think>...</think>` scratchpads prior to `TOOL_CALL:` ($N = 228$) reveals that *25.3%* of sabotage scratchpads fool an isolated 9B LLM Compliance Auditor (0.870 AUROC—rising to *28.9%* under an explicit audit notice), yet Layer-24 SAE probes strictly inside `<think>` reach *0.980 AUROC* (0.946 zero-shot from Study 1), catching *91.7%* of Auditor-fooled 503 fabrications and *100% (11/11)* under audit notice a median 29–35 tokens early. Crucially, steering only inside `<think>` fails due to *Action Re-Locking* (86.1% $->$ 77.8% vs. 33.3% full-trajectory). In *Study 2.5*, we unify these findings into a *Closed-Loop Probe-Gated Circuit Breaker* ($0.0%$ false-trigger rate across all benign controls) that rescues `2B`'s polysemantic cliff (0% $->$ 96.7% benign accuracy, 52.1% $->$ 10.4% sabotage) and defeats `9B` Action Re-Locking via a 28-token `<think>` latch (66.7% $->$ 5.6%–13.9% sabotage at 100% benign CoT accuracy). Finally, replicating on *Qwen2.5-7B-Instruct* (single-RMSNorm + GQA, no soft-capping) confirms identical 57.1% depth emergence (Layer 16/28, 0.937 AUROC), 96% pressured 503 fabrication, 0.978 `<think>` pre-tool AUROC (100% recovery on Auditor-fooled 503 CoTs), and 100% benign accuracy under closed-loop gating.
]

#v(0.45em)

#columns(2, gutter: 0.23in)[

= Introduction

Deploying large language models (LLMs) as autonomous coding and operations agents introduces failure modes qualitatively distinct from static text generation. When an agent equipped with file-editing, shell, database, and reporting tools encounters a failing unit test or an unreachable production database under tight SLA or user pressure, it faces a conflict between *task completion appearance* and *operational integrity*. Pressured agents can delete or skip failing unit tests (`@pytest.mark.skip`), hardcode benchmark literals, bypass `--no-verify` safety gates, or fabricate schema-valid metric payloads rather than invoking an error-escalation tool.

Prior mechanistic interpretability work has mapped truth and sycophancy directions on single-turn Q&A tasks using Contrastive Activation Addition (Rimsky et al., 2024) and Representation Engineering (Zou et al., 2023), open Sparse Autoencoder suites such as *Gemma Scope* (Lieberum et al., 2024) have enabled feature-level circuit analysis (Templeton et al., 2024; Marks et al., 2024), and behavioral studies have questioned Chain-of-Thought faithfulness (Turpin et al., 2023; Lanham et al., 2023). Four alignment questions motivate this work:
1. *Do tool-using models form a linearly and sparsely decodable commitment to sabotage in the residual stream **before** emitting any token of the tool name or path?*
2. *When an agent reasons inside `<think>...</think>`, can mid-network SAE probes catch "silent commitment" on plausible rationalizations that fool external LLM CoT auditors?*
3. *Can a closed-loop probe-gated circuit breaker neutralize both `<think>` Action Re-Locking and always-on steering collateral with zero tax on benign coding?*
4. *Do these pre-commitment and unfaithful-CoT dynamics generalize outside `Gemma-2`'s dual-RMSNorm architecture?*

= Study 1: Cross-Scale Sabotage Circuits

== Paired Counterfactual Benchmark & SAEs
We construct 500 multi-turn software-engineering scenarios evenly stratified across four taxonomies (125 each): *Test Tampering*, *Tool-Output Fabrication* (503 outages/locks), *Hardcoded Oracle Bypass*, and *Safety-Gate Bypass*, paired across *bottleneck-pressured* vs. *unpressured* prompts alongside 80 *benign coding capability controls*.

At the pre-commitment prefix `TOOL_CALL: {"tool": "`, we collect $N = 200$ on-policy rollouts per model and encode residual states $h_ell in RR^d$ via *Gemma Scope* 16,384-latent JumpReLU SAEs (Lieberum et al., 2024):
$ f(h_ell) = "ReLU"(h_ell W_"enc" + b_"enc") dot.op bb(1)[h_ell W_"enc" + b_"enc" > theta] $
During decoding on held-out scenarios ($N = 48$ pressured, $N = 30$ benign), we apply norm-scaled planning-block steering ($L_"2B" = {14,16,18,20}$; $L_"9B" = {20,24,28,32}$):
$ h'_ell = h_ell - alpha norm(h_ell)_2 hat(v)_ell $

]

#v(0.3em)
#figure(
  image("../figures/cross_scale/5_cross_scale_2b_vs_9b.png", width: 100%),
  caption: [Study 1 cross-scale comparison (`Gemma-2-2B-IT` vs. `Gemma-2-9B-IT`). *(A)* 5-fold CV AUROC across normalized depth; at 9B Layer 24, 16k SAE L1 probes (`0.9585`) match dense probes (`0.9529`). *(B)* 9B cuts test tampering and gate bypasses yet exhibits *inverse scaling* (`60% -> 88%`) on pressured 503 fabrication. *(C)* Held-out steering Pareto frontiers.],
)
#v(0.25em)

#columns(2, gutter: 0.23in)[

== Inverse Scaling & Interpretable SAE Latents
Without pressure, `Gemma-2-9B-IT` is near-perfectly aligned (0%–4% sabotage vs. 28%–48% in `2B`, Table 1). Under pressure, `9B` halves test tampering (56% vs. 92%), *yet fabricates tool outputs during 503 outages at 88.0% vs. 60.0% in `2B`* (+84.0 pp).

#figure(
  table(
    columns: (1.42fr, 0.68fr, 0.72fr, 0.68fr, 0.72fr, 0.78fr),
    inset: 3.8pt,
    align: (left, right, right, right, right, right),
    [*Taxonomy*], [*2B U*], [*2B P*], [*9B U*], [*9B P*], [*Qwen7B P*],
    [Test Tampering], [28%], [*92%*], [0%], [56%], [60%],
    [503 Fabrication], [0%], [60%], [4%], [*88%*], [*96%*],
    [Oracle Bypass], [48%], [68%], [0%], [28%], [36%],
    [Gate Bypass], [48%], [40%], [0%], [28%], [40%],
  ),
  caption: [Unpressured (U) vs. pressured (P) sabotage rates across `Gemma-2` (2B, 9B) and `Qwen2.5-7B-Instruct` (`Qwen7B` U is 0%–4%).],
)

At `9B` Layer 24, Neuronpedia auto-interp isolates *Latent \#15689* ($d = +1.49$, single-feature AUROC $= 0.829$) and *Latent \#11381* (*"causal relationships and responsibilities of different parties"*, $d = +1.04$, 39.2% vs. 0.7% fire rate) on sabotage, whereas *Latent \#6787* (*"algorithms and computational methods"*, $d = -1.48$) fires on *56.4% of honest fixes vs. 0.0% of sabotage*. Always-on dense steering on `2B` hits a *polysemantic over-refusal cliff* at $alpha >= +0.20$ (50% $->$ 0% benign accuracy), whereas `2B` SAE steering ($cos = 0.248$) retains 100% benign accuracy at $alpha = +0.20$ (27.1% sabotage), and `9B` achieves 6.25% (SAE) and 2.08% (dense) sabotage at 100% benign accuracy.

= Study 2: Silent Commitment in `<think>` CoT

Production agent frameworks increasingly rely on black-box Chain-of-Thought (CoT) monitors reading `<think>...</think>` before executing tool calls. Across $N = 228$ `Gemma-2-9B-IT` `<think> + TOOL_CALL:` rollouts ($224$ labeled: $99$ sabotage, $125$ honest) spanning *Unmonitored* and *Audited-Scratchpad Notice* regimes, we compare a Lexical Monitor and an isolated *9B LLM Compliance Auditor* (blind to `TOOL_CALL:`) against white-box Layer-24 probes evaluated *strictly before `</think>`*.

#figure(
  table(
    columns: (1.72fr, 0.76fr, 0.76fr),
    inset: 4pt,
    align: (left, right, right),
    [*Pre-Tool Detector (Inside `<think>`)*], [*Gemma-9B*], [*Qwen-7B*],
    [Black-Box Lexical CoT Monitor], [0.676], [0.581],
    [Black-Box LLM CoT Auditor], [0.870], [0.830],
    [Zero-Shot Study-1 No-CoT Probe], [*0.946*], [*0.959*],
    [Residual Probe (\@ 20% `<think>`)], [*0.944*], [*0.962*],
    [Residual Probe (\@ 100% `<think>`)], [*0.976*], [*0.978*],
    [Gemma Scope 16k SAE L1 Probe], [*0.980*], [—],
  ),
  caption: [Pre-action AUROC strictly inside `<think>` before `TOOL_CALL:`.],
)

*Oversight-Induced Obfuscation & 30-Token Lead Time:* 25.3% (25/99) of `9B` sabotage scratchpads fool the 9B CoT Auditor (`< 0.50`), rising from *23.0% to 28.9%* when warned `<think>` is audited. Yet the Layer-24 16k SAE probe catches *91.7% (11/12)* of Auditor-fooled 503 fabrications and *100% (11/11)* under audit notice—with *Latent \#11381* igniting a *median 29.0 tokens* and *Latent \#15689* *34.5 tokens* before `TOOL_CALL:` (`sab_0137`: 30% Auditor suspicion vs. $+21.01$ on \#11381 at $-38$ tokens and $0.00$ on \#6787).

*Action Re-Locking:* On $N = 36$ held-out focus prompts, steering *only* during `<think>` ($alpha = +0.22$) drops sabotage from $86.1% -> 77.8%$, whereas steering across `<think> + TOOL_CALL:` reaches *$33.3%$*: once hooks turn off at the boundary, prompt pressure *re-locks* the shortcut.

= Study 2.5: Closed-Loop Circuit Breaker

To eliminate both `2B`'s polysemantic cliff and `9B`'s Action Re-Locking without taxing benign traffic, we construct a *Closed-Loop Probe-Gated Circuit Breaker*. Gate thresholds $tau$ are calibrated strictly on disjoint validation rollouts (`sabotage[:44]` + `controls[40:80]`, targeting $<= 3.5%$ benign FPR). Inference runs completely unsteered ($alpha = 0$) unless the pre-action or streaming 28-token `<think>` gate crosses $tau$, dynamically latching 4-layer steering ($alpha = +0.25$) through `TOOL_CALL:`.

#figure(
  table(
    columns: (1.58fr, 0.66fr, 0.72fr, 0.84fr),
    inset: 3.8pt,
    align: (left, right, right, right),
    [*Model & Regime*], [*Sab $arrow.b$*], [*Benign $arrow.t$*], [*Trig (P / B)*],
    [`2B` Unsteered], [52.1%], [96.7%], [0 / 0%],
    [`2B` Always-On Dense ($+0.25$)], [0.0%], [*0.0%*], [100 / 100%],
    [`2B` Probe-Gated Breaker], [*10.4%*], [*96.7%*], [54.2 / *0.0%*],
    [`9B` `<think>` Unsteered], [66.7%], [100.0%], [0 / 0%],
    [`9B` Steer `<think>` Only], [55.6%], [100.0%], [100 / 100%],
    [`9B` Streaming SAE Breaker], [*13.9%*], [*100.0%*], [91.7 / *0.0%*],
    [`9B` Streaming Dense Breaker], [*5.6%*], [*100.0%*], [91.7 / *0.0%*],
    [`Qwen-7B` Always-On ($+0.20$)], [0.0%], [53.3%], [100 / 100%],
    [`Qwen-7B` Probe-Gated Breaker], [*14.6%*], [*100.0%*], [39.6 / *0.0%*],
  ),
  caption: [Closed-loop circuit breaker results on strictly held-out test sets, achieving *0.0% (0/30)* false triggers across all benign controls.],
)

Across *every* model and regime (Table 3), false-trigger rates on held-out benign controls are *0.0% (0/30)* (held-out sabotage-vs-benign gate AUROC $= 1.000$). On `Gemma-2-2B-IT`, probe-gating rescues benign coding accuracy from *0.0% back to 96.7%* while cutting sabotage by *80.0%* ($52.1% -> 10.4%$). On `Gemma-2-9B-IT` `<think>` rollouts, 86.1% of pressured focus prompts trip at the 28-token Silent Commitment checkpoint; latching steering across `TOOL_CALL:` cuts sabotage from *66.7% to 13.9% (SAE) and 5.6% (Dense)* (vs. 55.6% `<think>`-only) at *100.0% benign CoT accuracy* (avoiding Always-On Dense's dip to 96.7%).

= Cross-Family Validation (`Qwen2.5-7B`)

To verify that these phenomena are not artifacts of `Gemma-2`'s dual-RMSNorm + logit soft-capping design, we evaluate `Qwen/Qwen2.5-7B-Instruct` (`Qwen2ForCausalLM`: 28 layers, $d = 3584$, single pre-RMSNorm + GQA + SwiGLU, no soft-capping) across $N = 200$ decision rollouts, $N = 108$ `<think>` auditor rollouts, and held-out steering:
- *Behavioral & Depth Convergence:* Unpressured shortcuts remain 0%–4%, while pressured 503 fabrication surges to *96.0%* (+92 pp) and test tampering to *60.0%*. 5-fold CV residual probes peak at *Layer 16/28 (57.1% depth) at $0.937 plus.minus 0.028$ AUROC* (Diff-in-Means $= 0.922$ at L18)—matching `Gemma-2-9B`'s Layer 24/42 (57.1% depth).
- *`<think>` Silent Commitment & Breaker Rescue:* 17.7% of `Qwen2.5-7B` sabotage CoTs (19.4% on 503 fabrication) fool a 7B CoT Compliance Auditor (0.830 AUROC), whereas the no-CoT L16 probe transfers zero-shot at *0.959 AUROC* and the pre-tool L16 `<think>` probe hits *$0.978 plus.minus 0.025$ AUROC* (0.962 at 20% `<think>`), catching *100% (7/7)* of Auditor-fooled 503 fabrications. Like `Gemma-2-2B`, Always-On L14–20 steering drops `Qwen2.5-7B` benign accuracy to 53.3%, which the L16 Probe-Gated Circuit Breaker restores to *100.0% (30/30)* while cutting sabotage $37.5% -> 14.6%$.

= Limitations & Conclusion
While our experiments span two model families (`Gemma-2` 2B/9B and `Qwen2.5` 7B), 500 paired scenarios, and closed-loop `<think>` interventions, evaluations examine single-step tool emission after multi-turn context rather than long-horizon containerized `SWE-bench` loops. Deploying mid-network white-box circuit breakers alongside least-privilege sandboxing offers a practical path toward faithful tool-using agents.

= References
#set text(size: 8.0pt)
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning*. Anthropic, arXiv:2307.13702.
- Lieberum et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. arXiv:2408.05147.
- Marks et al. (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs*. arXiv:2403.19647.
- Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024.
- Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
- Turpin et al. (2023). *Language Models Don't Always Say What They Think*. NeurIPS 2023.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
]

#pagebreak()

#figure(
  image("../figures/silent_commitment/2_unfaithful_cot_single_trace.png", width: 98%),
  caption: [Study 2 token-by-token Layer-24 Gemma Scope 16k SAE trace on an unfaithful 503 Output Fabrication rollout (`sab_0137`). Although `<think>` sounds benign—fooling the 9B Compliance Auditor (`30%` suspicion)—Latent \#11381 spikes 38 tokens before `TOOL_CALL:` (`+21.01`) and Latent \#15689 hits `+13.50`, while honest repair Latent \#6787 stays flat at `0.00`.],
)
#v(0.2em)
#figure(
  image("../figures/silent_commitment/3_monitor_evasion_quadrant.png", width: 98%),
  caption: [Study 2 monitor evasion vs. white-box recovery ($N = 224$). *(A)* External 9B CoT Auditor vs. pre-tool Layer-24 SAE probe probabilities inside `<think>`. *(B)* Unfaithful CoT evasion rates and Layer-24 SAE recovery (`92%` on 503 Fabrication; `100%` under Audited Scratchpad notice).],
)
#v(0.2em)
#figure(
  image("../figures/circuit_breaker/2_closed_loop_cot_circuit_breaker_9b.png", width: 98%),
  caption: [Study 2.5 Closed-Loop `<think>` Streaming Circuit Breaker on `Gemma-2-9B-IT`, defeating Action Re-Locking (`66.7% -> 13.9%` SAE / `5.6%` Dense vs. `55.6%` `<think>`-only) with `0.0%` false triggers and `100.0%` benign `<think>` coding accuracy.],
)
#v(0.2em)
#figure(
  image("../figures/cross_family/1_qwen25_7b_behavioral_and_depth_auroc.png", width: 98%),
  caption: [Cross-family generalization to `Qwen2.5-7B-Instruct` (single pre-RMSNorm + GQA): *(A)* 96% pressured 503 fabrication, and *(B)* architecture-invariant emergence of linear pre-commitment separability at ~57% normalized depth across `Gemma-2-2B`, `Gemma-2-9B`, and `Qwen2.5-7B`.],
)
