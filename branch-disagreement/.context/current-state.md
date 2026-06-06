# Current State

## Current known implementation state

- Project-memory scaffolding created (AGENTS/CODEX/CLAUDE, `.context/`, `docs/`).
- `prototype/` contains the experiment harness:
  - dataset loaders normalising PopQA / TriviaQA into a common schema,
  - a model-runner abstraction with two implementations: a dependency-free
    **proxy** runner for CPU smoke tests, and a **vLLM** runner for the GPU box,
  - shared-prefix branch sampling (self-consistency and localised-marker modes),
  - NLI-based semantic clustering and a deterministic exact-match fallback,
  - disagreement scores plus baselines (token logprob, lexical agreement,
    semantic entropy),
  - pure-stdlib metrics: AUROC with bootstrap CI and a DeLong test.
- `deploy/` contains the vast.ai SSH toolkit (deploy, remote setup, run, fetch).

## Recently completed work

- Stood up the project from the localised-reasoning "Idea 2" discussion.
- Implemented the full pipeline end to end in proxy mode so it runs on CPU with
  no heavy dependencies.

## First real GPU results (2026-06-01) — see experiments/2026-06-01-popqa-and-triviaqa.md

Ran Qwen2.5-14B-AWQ under vLLM on a vast.ai RTX 4090, two 500-question runs.

- **Concept validated**: disagreement predicts hallucination, AUROC 0.72–0.75,
  tight CIs, both short-form (PopQA) and long-form (TriviaQA).
- **Semantic entropy > lexical disagreement in long-form** (DeLong p=0.002 /
  0.031) — semantic clustering earns its keep when answers are verbose. Clean,
  significant, literature-consistent.
- **Headline thesis FALSIFIED**: a SINGLE generation's logprob (`neg_logprob_single`,
  1/8 the cost) is statistically tied with everything — 8-branch disagreement,
  8-branch logprob, and semantic entropy (DeLong p=0.80 long-form). Multi-branch
  disagreement buys nothing over one sample's confidence on this white-box model.
- **Remaining niche**: disagreement only wins where logprobs are unavailable
  (black-box closed APIs). That, not "cheap detection," is the honest future framing.
- **STEERING CHANGES IT (Run E, RTX 5090)**: with `localised` steered-perspective
  branches (not temperature noise), 8-branch confidence SIGNIFICANTLY beats a
  single sample in long-form (0.784 vs 0.730, DeLong p=0.002) and recovers answers
  3:1 (+1.8pp) vs self-consistency's wash. Multiple branches are only worth their
  cost when STEERED. Caveat: crude markers cost ~3-4pp raw accuracy (steered
  majority 0.630 < unsteered single 0.654); better markers + a confidence-weighted
  selector are the path to an absolute win. This validates the core
  localised-reasoning intuition and is the most promising direction.
- **Run F (selector + redesigned markers) — mixed**: confidence-weighted selector
  helps marginally (conf>majority>single, ~1pp). Steered 8-branch confidence still
  significantly beats single (0.772 vs 0.727, p=0.014) — durable detection win.
  BUT the redesigned "truth-preserving" markers REGRESSED disagreement (branch
  0.761->0.687); generic markers were better — marker design is empirical.
  Absolute-accuracy win NOT achieved: best steered (0.644) still < unsteered single
  (0.652). The ~2pp steering perturbation tax persists; no marker set avoids it.
  Open crux: markers that diverge in reasoning without degrading the answer.
- **Run G (FUSED detectors) — best detection result**: combining single-sample
  logprob + semantic_entropy (parameter-free z-sum) is the best detector found,
  0.755, significantly beating single alone (p=0.027) and edging the 8-branch
  logprob — the Run D complementarity cashed out. Modest (~0.02) and
  mode-dependent (self-consistency only); p=0.027 is suggestive under multiple
  comparisons. Best practical detector = FUSION, pairs with logprob-gating.
- **Run H (selective prediction, n=500 then n=2000) — corrects Run G**: the
  noticeable-value framing is risk-coverage (answer when confident). At n=500
  fusion looked +3.2pp@50%cov, but at n=2000 it shrank to +0.6pp (ns) — the n=500
  fusion edge was mostly small-sample noise. DURABLE result: selective answering
  itself is the real value (base acc 0.698 -> ~0.85 @50%cov, ~0.89 @30%cov), and
  semantic_entropy is the BEST selective detector (acc@30%=0.895 > logprob).
  Always scale n before believing a fusion/selector delta.

GPU-environment fixes baked in: `VLLM_USE_FLASHINFER_SAMPLER=0` (bare image lacks
curand.h), per-branch seeds (a shared seed made all branches identical),
chat-template prompts, `mandarjoshi/trivia_qa` dataset id, response_mode short/long.

## Active gaps

- The fair cost comparison (single-sample logprob, 1/8 the generations) is NOT
  run yet — that's the test that decides the cost thesis. Current
  `neg_mean_logprob` averages over 8 branches, so it isn't actually cheaper.
- Long-form NLI clusters over full generations and over-splits paraphrases;
  answer-extraction clustering not yet implemented.
- Containment labeling over verbose text can over-credit correctness.

## Next likely tasks

- Single-sample logprob baseline → the real AUROC-vs-cost frontier.
- Answer-extraction ("Final answer: X") for clustering + labeling.
- localised vs self-consistency at equal budget; N-branches sweep (2/4/8/16).

## Known risks or fragile areas

- vLLM / model / NLI versions on the rented box may drift from local assumptions;
  `deploy/vast_setup.sh` pins what it can but the base image varies by host.
- Short-answer correctness scoring is normalization-based; ambiguous gold answers
  can mislabel correctness and distort AUROC.
