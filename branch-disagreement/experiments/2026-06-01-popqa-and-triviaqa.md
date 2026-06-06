# 2026-06-01 — First real GPU runs (Qwen2.5-14B-AWQ on a vast.ai RTX 4090)

Model: `Qwen/Qwen2.5-14B-Instruct-AWQ` under vLLM 0.22 (FlashInfer sampler
disabled — bare image lacked `curand.h`). NLI clusterer:
`microsoft/deberta-large-mnli`. 8 branches, self-consistency mode, temperature
0.8, per-branch seeds for diversity. Positive class = error (answer wrong).
Total GPU cost for the whole session: a few cents.

## Run A — short-form, PopQA (500 questions)

terse one-line answers, `stop=["\n"]`, max 32 tokens, exact-match labeling.

- accuracy 0.262 · 23 generated tokens/question · 31.5s generation

| detector | AUROC | 95% CI | DeLong p vs branch |
| --- | --- | --- | --- |
| branch_disagreement | 0.672 | [0.623, 0.720] | — |
| semantic_entropy | 0.674 | [0.624, 0.723] | 0.603 |
| lexical_disagreement | 0.675 | [0.627, 0.726] | 0.680 |
| neg_mean_logprob | 0.692 | [0.637, 0.746] | 0.198 |

Reads: all detectors above chance (concept validated). Semantic clustering adds
**nothing** over exact match (one-word answers have no synonyms to merge:
branch ≈ semantic ≈ lexical). Naive token-logprob is numerically best; no
detector significantly beats another.

## Run B — long-form, TriviaQA validation (500 questions)

free-form sentence answers, `stop=["\n\n"]`, max 96 tokens, gold-containment
labeling, NLI clustering over full generations.

- accuracy 0.658 · 305 generated tokens/question · 309s generation

| detector | AUROC | 95% CI | DeLong p vs branch |
| --- | --- | --- | --- |
| branch_disagreement | 0.720 | [0.677, 0.766] | — |
| semantic_entropy | 0.736 | [0.693, 0.783] | 0.002 (sig. better) |
| lexical_disagreement | 0.683 | [0.636, 0.726] | 0.031 (sig. worse) |
| neg_mean_logprob | 0.745 | [0.701, 0.790] | 0.172 (tie) |

Reads:
- Detection is easier in long-form (all detectors up ~0.05–0.07).
- **Semantic clustering now earns its keep**: semantic_entropy significantly
  beats raw branch_disagreement (p=0.002), and branch_disagreement significantly
  beats lexical_disagreement (p=0.031). Exact-match degrades on verbose answers
  exactly as predicted; NLI recovers the signal. This is the clean,
  literature-consistent effect.
- **But the headline thesis is not supported.** `neg_mean_logprob` is
  numerically best in BOTH regimes and is never significantly beaten. Token
  confidence is as good as disagreement.

## Run C — the decisive cost test: single-sample logprob (added to both runs)

Added `neg_logprob_single` = mean token logprob of ONE generation (branch 0),
1/8 the cost of the disagreement detectors. Same 500 questions, same labels.

| detector | short AUROC | long AUROC |
| --- | --- | --- |
| branch_disagreement | 0.672 | 0.720 |
| semantic_entropy | 0.674 | 0.736 |
| neg_mean_logprob (8) | 0.692 | 0.745 |
| **neg_logprob_single (1)** | **0.693** | **0.731** |

Exact pairwise DeLong, long-form, vs `neg_logprob_single`:

| vs | p-value |
| --- | --- |
| semantic_entropy | 0.797 |
| neg_mean_logprob (8) | 0.219 |
| branch_disagreement | 0.560 |

**Nothing significantly beats one generation's logprob** — not 8-branch
disagreement, not 8-branch averaged logprob, not semantic entropy. In short-form
the single-sample logprob is the single best detector outright.

## Honest conclusion

1. Core concept — disagreement/uncertainty predicts hallucination — is robustly
   validated (AUROC 0.72–0.75, tight CIs, both regimes).
2. Semantic entropy > lexical disagreement in long-form is a real, significant,
   reproducible effect. The "semantic clustering matters when answers are
   verbose" claim holds.
3. The "cheap disagreement beats naive confidence" claim is FALSIFIED here.
   A single generation's logprob is statistically tied with everything, including
   8-branch semantic entropy (DeLong p=0.80). Spending 8x compute on disagreement
   buys nothing over reading one sample's token probability.

## Run D — retrospective: logprob as a GATE, disagreement as the residual

Reframe: don't compete logprob vs disagreement — use logprob to route adaptive
compute (branch only on the low-confidence tail) and ask whether disagreement
catches the errors logprob is blind to (confident hallucinations).

Long-form TriviaQA (500q, base error 0.34):

- **Gate works**: least-confident 50% holds 123/171 errors (72%) at 0.49 error
  rate; most-confident 50% only 0.19. Branching the low-confidence tail catches
  most errors at a fraction of the compute.
- **Complementarity (within the most-confident 50%, AUROC vs error):**

  | detector | AUROC |
  | --- | --- |
  | neg_logprob_single | 0.605 |
  | neg_mean_logprob | 0.624 |
  | branch_disagreement | 0.654 |
  | semantic_entropy | 0.667 |

  The ranking INVERTS: once conditioned on confidence, disagreement separates
  right from wrong better than logprob. Suggestive (n=250, 48 errors, wide
  overlapping CIs), not conclusive — but the direction is the opposite of the
  unconditional result, evidence that disagreement catches confident-wrong cases
  logprob misses. This is the structure worth pursuing.

## Run E — STEERED (localised) branches vs self-consistency (the key result)

Re-ran both regimes with `branch_mode=localised` (each branch gets a distinct
perspective marker) vs `self_consistency` (same prompt, temperature noise).
Long-form TriviaQA, 500q, on an RTX 5090.

Long-form detection AUROC:

| detector | self-consistency | localised (steered) |
| --- | --- | --- |
| branch_disagreement | 0.720 | 0.761 |
| semantic_entropy | 0.736 | 0.768 |
| neg_mean_logprob (8) | 0.745 | 0.784 |
| neg_logprob_single (1) | 0.731 | 0.730 |

Decisive contrasts (DeLong + recovery):

| | self-consistency | localised (steered) |
| --- | --- | --- |
| 8-branch conf vs 1-sample conf | 0.745 vs 0.731, p=0.219 (tie) | 0.784 vs 0.730, **p=0.002 (sig)** |
| recovery (majority vs single acc) | +0.4pp, 11 rec / 9 reg (wash) | **+1.8pp, 13 rec / 4 reg (3:1)** |

**Steering unlocks the value of multiple branches.** With temperature noise, 8
branches don't significantly beat 1 and barely recover. With steered
perspectives, 8 branches significantly beat 1 (p=0.002) and recover at 3:1. This
validates the core localised-reasoning intuition: systematic perspective
divergence is a stronger signal than sampling noise.

**Honest caveat — steering costs raw accuracy.** The markers perturb the model:
single-sample acc 0.654 -> 0.612, majority 0.658 -> 0.630. So steered majority
(0.630) is still BELOW unsteered single (0.654) in absolute accuracy. Steering
makes branches more *informative* (detection, relative recovery) but slightly
degrades the answer. The markers used are generic ("Answer directly", "Recall the
source", ...) and the selector is naive majority. Better-designed markers + a
confidence/agreement-weighted selector are the path to convert the relative gain
into an absolute accuracy win. Short-form showed no steering benefit (one-token
answers leave nothing to steer).

## Run F — confidence-weighted selector + redesigned markers (mixed)

Goal: turn the relative steering win into an absolute accuracy win, via (a)
"truth-preserving" perspective markers and (b) a confidence-weighted selector
(pick the semantic cluster with the most exp(logprob) mass). Long-form, 500q,
RTX 5090, per-branch logprobs+clusters logged so selectors compare offline.

Absolute accuracy by selector:

| mode | single | majority | conf-weighted |
| --- | --- | --- | --- |
| self-consistency | 0.652 | 0.658 | 0.660 |
| localised (steered) | 0.628 | 0.636 | 0.644 |

Detection AUROC:

| detector | SC | localised |
| --- | --- | --- |
| branch_disagreement | 0.715 | 0.687 |
| semantic_entropy | 0.732 | 0.680 |
| neg_mean_logprob (8) | 0.745 | 0.772 |
| neg_logprob_single (1) | 0.733 | 0.727 |

Findings:
1. **Confidence-weighted selector helps, but little**: conf > majority > single
   within each mode (~0.5-1.5pp). Right direction, small effect.
2. **The redesigned markers REGRESSED disagreement detection**: branch 0.761 ->
   0.687, semantic_entropy 0.768 -> 0.680 vs the earlier generic markers. The
   "more principled" markers were worse. Marker design is empirical; this
   intuition was wrong.
3. **The robust win survives**: steered 8-branch confidence still significantly
   beats single (0.772 vs 0.727, DeLong p=0.014; SC p=0.262). Steering's value
   shows up through confidence aggregation, reproducibly.
4. **Absolute-accuracy vindication NOT achieved**: best steered (conf-weighted
   0.644) is still BELOW unsteered single (0.652), -0.8pp. The ~2pp steering
   perturbation tax persists; no marker set tried avoids it.

Net: the durable, reproducible result is that **steered branching is a better
hallucination DETECTOR than a single sample** (significant, both marker sets). It
is not yet a better ANSWERER — steering costs accuracy that better selection
doesn't recover. An absolute win needs markers that diverge in reasoning without
degrading the answer; none tried so far manage that.

## Run G — FUSED detectors (logprob + disagreement together)

Tested whether combining signals beats either alone — the payoff of Run D's
complementarity. Parameter-free equal-weight z-score sum (no fitting => no
overfit). Long-form 500q, offline from the detail CSVs.

Self-consistency:

| detector | AUROC | DeLong p vs best component |
| --- | --- | --- |
| single logprob | 0.733 | |
| 8-branch logprob | 0.745 | |
| semantic_entropy | 0.732 | |
| **single + semantic_entropy** | **0.755** | **0.027** (vs single) |
| single + branch_disagreement | 0.751 | 0.070 (vs single) |
| 8-branch logprob + semantic_entropy | 0.757 | 0.160 (vs logp8) |

Localised: fusion did NOT significantly help (logp8 already strong at 0.772;
disagreement weak at 0.687, little complementary room).

Findings:
- **Fusing cheap single-sample confidence + semantic entropy is the best detector
  found (0.755), significantly beating single alone (p=0.027) and edging past the
  8-branch logprob.** The complementarity (Run D) cashes out: the two signals
  catch different errors.
- Effect is modest (~0.02 AUROC) and mode-dependent (helps self-consistency, not
  localised).
- Rigor caveat: 8 combos tested; under strict multiple-comparison correction
  (Bonferroni ~0.006) p=0.027 is suggestive, not definitive. Worth a confirmatory
  run, but the direction matches the Run D mechanism.

Practical upshot: the best hallucination detector is neither logprob nor
disagreement alone but their FUSION — and it pairs naturally with logprob-gating
(cheap route, fused score as the better second-stage check).

## Run H — selective prediction (risk-coverage): the value made visible

Insight: a global AUROC averages over the easy confident questions, hiding the
detector's value. The deployment-relevant metric is selective prediction —
answer when confident, abstain/escalate when not. Rank by detector (most
confident first), retain the top X% (coverage), measure accuracy on retained.
Offline, self-consistency long-form, n=500, base accuracy 0.658.

| detector | acc@50%cov | acc@30%cov | cov@90%acc |
| --- | --- | --- | --- |
| single-logprob | 0.808 | 0.840 | 0.14 |
| 8-branch logprob | 0.816 | 0.860 | 0.13 |
| semantic_entropy | 0.812 | 0.847 | 0.20 |
| FUSION(single+se) | 0.840 | 0.880 | 0.17 |

- The +0.02 AUROC fusion edge becomes a tangible **+3.2pp at 50% coverage / +4pp
  at 30%**: answer the confident half and fusion is right 84.0% vs 80.8% for
  logprob alone. Selective prediction is how to *show* the value.
- Significance at n=500: fusion - single @50%cov = +3.2pp, 95% CI [-0.8,+5.2],
  P(fusion>single)=0.905. Suggestive, not yet significant — 250 retained points
  is underpowered. Hence the n=2000 confirmatory run.
- semantic_entropy alone wins the high-precision corner (cov@90%acc=0.20): more
  questions answerable at >=90% accuracy. Different detectors own different
  operating points.

This reframes the deliverable: not "AUROC 0.755" but a **risk-coverage / selective
answering** story.

### Run H confirmation at n=2000 (corrects the n=500 fusion claim)

Scaled to n=2000 (base accuracy 0.698) to power the test. Selective accuracy:

| detector | acc@50% | acc@30% | acc@20% |
| --- | --- | --- | --- |
| single-logprob | 0.850 | 0.873 | 0.868 |
| 8-branch logprob | 0.855 | 0.877 | 0.890 |
| semantic_entropy | 0.858 | **0.895** | 0.892 |
| FUSION(single+se) | 0.856 | 0.890 | 0.912 |

- **The fusion advantage largely evaporated with more data.** fusion - single:
  +0.6pp @50%cov (CI [-0.5,+2.4], ns) and +1.7pp @30%cov (CI [-0.2,+3.5],
  P=0.95, borderline). The n=500 +3.2pp was mostly small-sample noise. Honest
  correction: fusion is NOT a robust winner over the best single detector.
- **What IS robust and noticeable: selective answering itself.** A model at 69.8%
  base accuracy answers its most-confident 50% at ~85%, and its most-confident
  30% at ~88-90%. That ~+20pp on the retained set is the real, deployable value
  of the uncertainty signal — common to logprob, semantic entropy, and 8-branch
  logprob.
- **semantic_entropy is the best selective detector** (acc@30%=0.895), beating
  single-logprob and even 8-branch logprob, despite logprob's slightly higher
  global AUROC — it concentrates discrimination in the high-confidence region
  that selective answering exploits.

Tight-CI AUROC at n=2000: neg_mean_logprob 0.760, semantic_entropy 0.751,
branch_disagreement 0.743 = single 0.743, lexical 0.694. semantic_entropy
significantly beats branch_disagreement (p=0.001); single ties branch (p=0.991).

## What this means for the project thesis

On a **white-box, well-calibrated open model**, multi-branch disagreement is not
worth its cost as a hallucination detector — single-sample confidence is as good.
The genuine remaining niche for disagreement / semantic entropy is the
**black-box** setting: closed APIs that expose no token logprobs, where the cheap
baseline does not exist and disagreement is the best available signal. That, not
"cheap detection," is the honest framing for any future work. Boundary conditions
worth testing before generalising: poorly-calibrated / overconfident RLHF models
(where logprob may fail while disagreement holds), and non-quantized weights (AWQ
may affect logprob calibration).

## Important caveats / threats to validity

- Our `neg_mean_logprob` averages over the 8 branches, so it is NOT cheaper than
  the disagreement methods in this implementation. The fair "cost" comparison
  (single-sample logprob, 1/8 the generations) has not been run yet — that is
  the test that would actually support or kill the cost thesis.
- Long-form NLI clusters over FULL generations, which over-splits paraphrases of
  the same correct answer (observed: correct answers with disagreement 0.875).
  Clustering on an extracted answer span should raise the disagreement/semantic
  scores further. Not yet implemented.
- Containment labeling over verbose text can over-credit correctness when gold
  alias lists are broad (TriviaQA). An answer-extraction or LLM-judge label would
  be more precise.

## Next experiments (in priority order)

1. **Single-sample logprob baseline** (1 generation) vs the 8-branch detectors —
   the real cost-vs-quality frontier. This decides the cost thesis.
2. **Answer-extraction** ("Final answer: X") so clustering + labeling operate on
   the answer, not the ramble — should lift semantic_entropy / branch_disagreement.
3. localised branch mode vs self-consistency at equal token budget.
4. N-branches sweep (2/4/8/16) — how few branches preserve the signal.
