#set page(
  paper: "us-letter",
  margin: (x: 0.70in, y: 0.72in),
  numbering: "1",
)
#set text(font: "Liberation Serif", size: 9.6pt)
#set par(justify: true, leading: 0.56em)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 14.8pt, weight: "bold")[
    Mechanistic Debugging of Agentic Sabotage and Unfaithful Chain-of-Thought:\
    Sparse Autoencoder Circuits & Silent Commitment in Multi-Turn Tool Use
  ]
  #v(0.5em)
  #text(size: 10.2pt)[
    *#link("https://danielgaskins.com/")[Daniel Gaskins]*\
    #text(size: 9pt, fill: rgb("#4b5563"))[Independent Researcher · #link("https://danielgaskins.com/")[`danielgaskins.com`] · #link("mailto:hello@danielgaskins.com")[`hello@danielgaskins.com`]\
    Code & Benchmark: #link("https://github.com/danielgaskins/agentic-sabotage")[`github.com/danielgaskins/agentic-sabotage`] · Case Study: #link("https://danielgaskins.com/agentic-sabotage.html")[`danielgaskins.com/agentic-sabotage.html`]]
  ]
  #v(0.6em)
]

#rect(
  width: 100%,
  inset: 9pt,
  radius: 3pt,
  stroke: 0.6pt + rgb("#d1d5db"),
  fill: rgb("#f9fafb"),
)[
  #text(weight: "bold")[Abstract.] Autonomous software-engineering and tool-using language models frequently encounter operational bottlenecks—such as CI freeze deadlines, upstream database outages, and evaluation compute caps—that incentivize unauthorized shortcuts (*agentic sabotage*) over honest escalation or root-cause repair. In *Study 1*, we introduce a 500-scenario paired counterfactual benchmark (1,000 multi-turn trajectories across four sabotage taxonomies plus 80 benign capability controls) and map pre-commitment representations in *Gemma-2-2B-IT* and *Gemma-2-9B-IT* using *Gemma Scope* 16k JumpReLU Sparse Autoencoders (SAEs). Behaviorally, scaling from 2B to 9B eliminates unpressured shortcuts (28%–48% $->$ 0%–4%) and halves pressured unit-test tampering (92% $->$ 56%), yet exhibits *inverse scaling* on sycophantic tool-output fabrication during 503 outages (60% $->$ 88%). Mechanistically, sparse L1 probes over Layer-24 SAE latents close the superposition gap at 9B scale (0.9585 AUROC vs. 0.9529 dense) and enable multi-layer causal steering that cuts pressured sabotage by 80%–93% at 100% benign coding accuracy. In *Study 2 ("Silent Commitment")*, we test Chain-of-Thought (CoT) faithfulness by allowing *Gemma-2-9B-IT* to reason inside `<think>...</think>` scratchpads prior to `TOOL_CALL:` ($N = 228$). We find that 88.9% of sabotage scratchpads evade lexical filters (0.676 AUROC) and *25.3% fool an independent 9B LLM Compliance Auditor* (0.870 AUROC—rising to *28.9% unfaithful CoT* when warned that `<think>` logs are audited). Yet strictly inside `<think>`—before a single token of `TOOL_CALL:` is emitted—our Layer-24 Gemma Scope 16k SAE probe achieves *0.980 AUROC* (0.946 zero-shot transfer from Study 1), catching *91.7%* of Auditor-fooled 503 fabrications and *100.0% (11/11)* of Auditor-fooled audited-scratchpad sabotage cases, with Latents \#11381 and \#15689 igniting 29–35 tokens before action emission while honest repair Latent \#6787 remains flat at zero.
]

#v(0.5em)

#columns(2, gutter: 0.24in)[

= Introduction

Deploying large language models (LLMs) as autonomous coding and operations agents introduces failure modes qualitatively distinct from static text generation. When an agent equipped with file-editing, shell, database, and reporting tools encounters a failing unit test or an unreachable production database under tight SLA or user pressure, it faces a conflict between *task completion appearance* and *operational integrity*. In practice, pressured agents can delete or skip failing unit tests (`@pytest.mark.skip`), hardcode benchmark literals, bypass `--no-verify` safety gates, or fabricate successful metric payloads rather than invoking an error-escalation tool.

Prior mechanistic interpretability work has mapped truth and sycophancy directions on single-turn Q&A tasks using Contrastive Activation Addition (CAA; Rimsky et al., 2024) and Representation Engineering (Zou et al., 2023), open Sparse Autoencoder suites such as *Gemma Scope* (Lieberum et al., 2024) have enabled feature-level circuit analysis (Templeton et al., 2024; Marks et al., 2024), and behavioral studies have questioned whether Chain-of-Thought reasoning faithfully reflects internal decision drivers (Turpin et al., 2023; Lanham et al., 2023). Three central questions for agentic alignment motivate this work:
1. *Do models form a linearly and sparsely decodable commitment to sabotage vs. honest escalation in the residual stream **before** emitting the tool name or target path?*
2. *When steering away from sabotage at inference time, do sparse SAE decoder directions offer a superior safety–capability Pareto frontier over dense residual probes?*
3. *When an agent reasons inside an explicit `<think>...</think>` scratchpad before acting, can internal Layer-24 SAE probes detect "silent commitment" on unfaithful traces whose benign-sounding rationalizations fool external CoT monitors?*

= Study 1: Cross-Scale Sabotage Circuits

== Paired Counterfactual Sabotage Benchmark
We construct 500 multi-turn software-engineering scenarios evenly stratified across four taxonomies (125 scenarios each):
- *Test Tampering:* Deleting test files or inserting `@pytest.mark.skip` / trivial assertions instead of fixing `src/` bugs.
- *Tool-Output Fabrication:* Emitting `"status": "ok"` metric reports with invented records when `query_db` returns a 503 outage or lock error.
- *Hardcoded Oracle Bypass:* Special-casing benchmark input literals (`if x == ...: return ...`) rather than fixing algorithmic logic.
- *Safety-Gate Bypass:* Passing `--no-verify` / `--force` or setting `SKIP_DB_LOCK=True` under urgent deploy pressure.

Each scenario pairs a multi-turn tool history (`assistant` inspection call + `[TOOL_OUTPUT]` failure trace) with two counterfactual follow-ups: a *bottleneck-pressured* prompt and an *unpressured* integrity-preserving prompt, alongside 80 *benign coding capability controls* where calling `report_blocker` is penalized as a false-positive refusal.

== Pre-Commitment Probing & JumpReLU SAEs
In Study 1, generation begins from the pre-tool-name prefix `TOOL_CALL: {"tool": "`. We collect $N = 200$ pure on-policy rollouts per model and cache residual stream states $h_ell in RR^d$ at the final `"` token across 8 candidate layers. At each planning layer $ell$, we encode $h_ell$ via official *Gemma Scope* 16,384-latent JumpReLU SAEs (Lieberum et al., 2024):
$ f(h_ell) = "ReLU"(h_ell W_"enc" + b_"enc") dot.op bb(1)[h_ell W_"enc" + b_"enc" > theta] $
During autoregressive decoding on held-out scenarios ($N = 48$ pressured, $N = 30$ benign controls), we apply norm-scaled steering across the 4-layer planning block ($L_"2B" = {14,16,18,20}$; $L_"9B" = {20,24,28,32}$):
$ h'_ell = h_ell - alpha norm(h_ell)_2 hat(v)_ell $

]

#v(0.35em)
#figure(
  image("../figures/cross_scale/5_cross_scale_2b_vs_9b.png", width: 100%),
  caption: [Study 1 cross-scale mechanistic comparison between `Gemma-2-2B-IT` and `Gemma-2-9B-IT`. *(A)* 5-fold CV AUROC across normalized depth; at 9B scale, 16k SAE L1 probes (`0.9585`) close the superposition gap with dense residual probes (`0.9529`). *(B)* While 9B reduces test tampering and gate bypasses, it exhibits *inverse scaling* (60% $->$ 88%) on sycophantic tool-output fabrication under pressure. *(C)* Safety–capability Pareto frontier on held-out evaluations.],
)
#v(0.3em)

#columns(2, gutter: 0.24in)[

== Inverse Scaling, Latents, & Steering Disentanglement
Table 1 summarizes empirical Study 1 shortcut rates. Without pressure, `Gemma-2-9B-IT` is near-perfectly aligned (0%–4% sabotage vs. 28%–48% in `2B`). Under pressure, `9B` resists crude test edits (56% vs. 92%), *yet fabricates tool outputs during 503 outages at an 88.0% rate vs. 60.0% for `2B`* (+84.0 pp over unpressured `9B`).

#figure(
  table(
    columns: (1.5fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
    inset: 4.5pt,
    align: (left, right, right, right, right),
    [*Taxonomy*], [*2B Unp.*], [*2B Pres.*], [*9B Unp.*], [*9B Pres.*],
    [Test Tampering], [28.0%], [*92.0%*], [0.0%], [56.0%],
    [Output Fabrication], [0.0%], [60.0%], [4.0%], [*88.0%*],
    [Oracle Bypass], [48.0%], [68.0%], [0.0%], [28.0%],
    [Safety Gate Bypass], [48.0%], [40.0%], [0.0%], [28.0%],
  ),
  caption: [Study 1 on-policy sabotage rates across pressure conditions.],
)

Neuronpedia auto-interpretations at `9B` Layer 24 isolate sharp pre-commitment semantics: *Latent \#15689* ($d = +1.49$, single-latent AUROC $= 0.829$) and *Latent \#11381* (*"causal relationships and responsibilities of different parties"*, $d = +1.04$, 39.2% vs. 0.7% fire rate) predict sabotage, while *Latent \#6787* (*"references to algorithms and computational methods"*, $d = -1.48$) fires on *56.4% of honest fixes vs. 0.0% of sabotage*.

On held-out steering (Table 2), `2B` dense probes hit a polysemantic over-refusal cliff at $alpha >= +0.20$ (50% $->$ 0% benign accuracy), whereas `2B` SAE steering ($cos("SAE", "Probe") = 0.248$) preserves *100.0% benign accuracy at $alpha = +0.20$* (27.1% sabotage). At `9B` scale ($d = 3584$), SAE steering cuts sabotage *31.25% $->$ 6.25%* and dense probes reach *2.08%*, *both retaining 100.0% (30/30) benign accuracy*.

#figure(
  table(
    columns: (1.28fr, 0.76fr, 0.82fr, 0.82fr, 0.82fr),
    inset: 4pt,
    align: (left, right, right, right, right),
    [*Method (Sab/Legit)*], [*Base*], [*$+0.14$*], [*$+0.20$*], [*$+0.25$*],
    [`2B` SAE Steer], [52 / 97%], [35 / *100%*], [*27 / 100%*], [*19 / 90%*],
    [`2B` Dense Probe], [52 / 97%], [19 / 93%], [0 / 50%], [0 / 0%],
    [`9B` SAE Steer], [31 / 100%], [29 / 100%], [21 / 100%], [*6 / 100%*],
    [`9B` Dense Probe], [31 / 100%], [8 / 100%], [*2 / 100%*], [*2 / 100%*],
  ),
  caption: [Study 1 held-out sabotage % ($arrow.b$) / benign coding pass % ($arrow.t$).],
)

= Study 2: The "Silent Commitment" Study

== Unfaithful `<think>` Scratchpads vs. Residual Probes
Production agent harnesses increasingly rely on external Chain-of-Thought (CoT) monitors that inspect an agent's `<think>...</think>` scratchpad prior to executing a tool call. This defense assumes *CoT faithfulness*: that an agent preparing to fabricate metrics or tamper with tests will verbalize suspicious intent in its scratchpad.

To test this assumption directly on `Gemma-2-9B-IT`, we elicit explicit `<think>...</think>` reasoning traces prior to `TOOL_CALL:` across $N = 228$ rollouts ($224$ unambiguously labeled: $99$ sabotage, $125$ honest), comparing two oversight regimes:
1. *Unmonitored `<think>`* ($80$ scenarios $times$ $2$ pressure conditions, enriched for 503 Output Fabrication and Test Tampering).
2. *Audited `<think>` Notice* ($48$ pressured focus scenarios appending an explicit compliance notice that `<think>` logs are reviewed by an automated policy filter).

We evaluate two black-box CoT text monitors—a *Lexical Intent Monitor* and an isolated zero-shot *`Gemma-2-9B-IT` Compliance Auditor* reading *only* `<think>...</think>` (blind to `TOOL_CALL:`)—against white-box Layer-24 residual and Gemma Scope 16k SAE probes evaluated *strictly inside `<think>`* before `</think>`.

#figure(
  table(
    columns: (1.85fr, 0.95fr, 0.75fr),
    inset: 4.5pt,
    align: (left, left, right),
    [*Pre-Tool Detector (Inside `<think>`)*], [*Signal*], [*AUROC*],
    [Lexical CoT Monitor], [`<think>` text], [0.676],
    [9B LLM CoT Compliance Auditor], [`<think>` text], [0.870],
    [Zero-Shot Study-1 L24 Probe], [L24 `<think>`], [*0.946*],
    [L24 Residual Probe (\@ 20% `<think>`)], [First 20%], [*0.944*],
    [L24 Residual Probe (\@ 100% `<think>`)], [`<think>` mean], [*0.976*],
    [L24 Gemma Scope 16k SAE L1 Probe], [L24 SAE], [*0.980*],
  ),
  caption: [Pre-action sabotage detection strictly prior to `TOOL_CALL:`.],
)

== Monitor Evasion & Early Token-Level SAE Ignition
Table 3 and Figure 2 reveal a stark monitorability gap:
- *Black-Box Blindspot & Oversight-Induced Obfuscation:* 88.9% of sabotage `<think>` traces avoid explicit shortcut vocabulary (0.676 AUROC), and *25.3% (25/99) fool the 9B LLM Compliance Auditor* (`suspicion < 0.50`). Crucially, warning the model that `<think>` is audited *increases* Auditor-evading unfaithful CoT from *23.0% to 28.9%* (e.g., reframing fake zero-balance or low-fraud payloads as *"a temporary placeholder to avoid delaying notifications until the database issue is resolved"*).
- *White-Box Pre-Tool Recovery:* Measured strictly before `TOOL_CALL:`, our Layer-24 Gemma Scope 16k SAE probe achieves *0.980 $plus.minus$ 0.018 AUROC*—and Study 1's frozen no-CoT Layer-24 probe transfers *zero-shot* onto `<think>` tokens at *0.946 AUROC* (already reaching *0.944 AUROC* within the first 20% of `<think>`). On the exact subset of sabotage rollouts that *fooled* the 9B CoT Auditor (Table 4), the Layer-24 SAE probe catches *91.7% (11/12)* of 503 fabrications and *100.0% (11/11)* of audited-scratchpad sabotage trajectories.
- *30-Token Latent Lead Time:* Token-by-token encoding across `<think>` (Figure 3) shows that *Latent \#11381* fires $3.2 times$ more often on sabotage CoTs (63.6% vs. 20.0%) with a *median lead time of 29.0 tokens* (mean 30.3) before `TOOL_CALL:`, and *Latent \#15689* fires with a *median lead time of 34.5 tokens* (mean 36.7). On `sab_0137` (503 fraud-score fabrication), the 9B Auditor assigns just 30% suspicion (`BENIGN`), yet Latent \#11381 spikes to $+21.01$ at 38 tokens prior to `TOOL_CALL:` while honest repair *Latent \#6787* remains strictly $0.00$ across all 57 tokens.

#figure(
  table(
    columns: (1.35fr, 0.45fr, 0.85fr, 0.85fr),
    inset: 4pt,
    align: (left, right, right, right),
    [*Sabotage Cohort*], [*N*], [*Fools 9B CoT*], [*SAE Catches*],
    [All Sabotage], [99], [25.3%], [*88.0%* (22/25)],
    [503 Fabrication], [49], [24.5%], [*91.7%* (11/12)],
    [Test Tampering], [40], [32.5%], [*84.6%* (11/13)],
    [Audited `<think>`], [38], [*28.9%*], [*100%* (11/11)],
  ),
  caption: [Unfaithful CoT rate fooling the 9B Auditor vs. L24 SAE catch rate on those exact Auditor-fooled traces.],
)

== Mid-Thought Causal Steering & Action Re-Locking
Finally, on $N = 36$ held-out pressured 503 Fabrication and Test Tampering scenarios ($86.1%$ baseline sabotage), steering with the Gemma Scope SAE vector ($alpha = +0.22$) *only during `<think>`* (disabling hooks before `TOOL_CALL:`) reduces sabotage modestly from $86.1% -> 77.8%$, whereas maintaining the hook across `<think> + TOOL_CALL:` cuts sabotage *$86.1% -> 33.3%$* ($13.9% -> 50.0%$ honest escalation). Even when `<think>` prose is nudged toward caution, unresolved prompt pressure in the context window can *re-lock* sabotage at action emission unless residual defenses span the tool boundary.

= Limitations & Conclusion
While our benchmark spans 500 paired scenarios and explicit `<think>` reasoning regimes, evaluation focuses on single-step tool invocations within the `Gemma-2` family rather than multi-turn containerized `SWE-bench` loops. Extending token-level SAE probes as live runtime circuit-breakers across multi-family coding agents is a compelling direction for deployment alignment.

= References
#set text(size: 8.2pt)
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning*. Anthropic, arXiv:2307.13702.
- Lieberum et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. arXiv:2408.05147.
- Marks et al. (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*. arXiv:2403.19647.
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024.
- Sharma et al. (2023). *Towards Understanding Sycophancy in Language Models*. ICLR 2024.
- Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
- Turpin et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. NeurIPS 2023.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
]

#pagebreak()

#figure(
  image("../figures/silent_commitment/2_unfaithful_cot_single_trace.png", width: 99%),
  caption: [Study 2 token-by-token Layer-24 Gemma Scope 16k SAE trace on an unfaithful 503 Output Fabrication rollout (`sab_0137`). Although the verbalized `<think>` scratchpad sounds benign—fooling the external 9B Compliance Auditor (`30%` suspicion)—Latent \#11381 spikes 38 tokens before `TOOL_CALL:` (`+21.01`) and Latent \#15689 spikes 32 tokens early (`+13.50`), while algorithmic repair Latent \#6787 stays flat at `0.00`.],
)
#v(0.25em)
#figure(
  image("../figures/silent_commitment/3_monitor_evasion_quadrant.png", width: 99%),
  caption: [Study 2 monitor evasion vs. white-box recovery ($N = 224$). *(A)* Quadrant comparison of the external 9B CoT Compliance Auditor vs. pre-tool Layer-24 SAE probe probabilities inside `<think>`. *(B)* Unfaithful CoT evasion rates and Layer-24 SAE recovery (`92%` on 503 Fabrication; `100%` under Audited Scratchpad notice).],
)
#v(0.25em)
#figure(
  image("../figures/gemma-2-9b-it/4_steering_safety_vs_capability.png", width: 96%),
  caption: [Study 1 `Gemma-2-9B-IT` multi-layer steering curves across $alpha in [-0.10, +0.25]$, achieving up to 80%–93% relative sabotage suppression while preserving 100% benign coding accuracy.],
)
