# Hidden-State Acceptance Probes for Speculative Decoding: A Decodability, Causality, and Systems Study

**Status:** internal research report, draft for review. Format: Markdown, not LaTeX
— this project has no target venue yet, the source of truth (`PROGRESS.md`) is
already Markdown, and the working relationship on this project has been iterative
editing rather than camera-ready formatting. If this is later aimed at a workshop
or conference, it should be re-typeset in LaTeX at that point, not before.

All results below are drawn directly from `PROGRESS.md`, the project's phase-by-phase
log, plus one supplementary analysis (round/position composition, `analysis/phase5/round_composition.json`)
run specifically to support the Phase 5 discussion. Nothing here re-derives or
reinterprets a result beyond what `PROGRESS.md` already established — numbers are
transcribed, not recomputed, except where explicitly marked as new.

---

## 1. Abstract

We test whether a target language model's intermediate hidden states, computed
during speculative decoding, contain a decodable signal for the exact
accept/reject decision speculative decoding's verification step would otherwise
require a full forward pass to compute. Using an independently pretrained
drafter (Qwen2.5-0.5B-Instruct) against a target (Qwen2.5-7B-Instruct) — a
decoupled pairing chosen specifically to avoid collapsing into self-speculative
decoding — we find the signal *is* decodable: a small MLP probe reads the
target's layer-24-of-28 residual stream and predicts the deterministic
acceptance-probability label with AUROC 0.86, peaking a few layers before the
final one rather than dramatically early, and is well-calibrated (ECE
0.01–0.03) where a comparable linear probe is not (ECE 0.25–0.28). A causal
check — mean-ablating the probe's read-out direction, under two independent
ablation designs (whole-sequence and position-restricted), each with a
matched random-direction control — finds **no evidence** that this direction
is causally load-bearing for the target's own verification computation: effect
sizes are indistinguishable from the control in aggregate and inconsistent
across domains. Proceeding anyway on "predictive is enough for a systems
shortcut" grounds, we build a verification-skipping gate using this probe and
measure it end-to-end: it is not lossless (5% of held-out completions diverge
from the exact algorithm's output even at a conservative threshold), and its
wall-clock effect is domain-dependent and mostly negative — a modest ~8%
median speedup in the code domain, offset by measured *slowdowns* in
reasoning and chat, driven by a diagnosed mechanism (redundant, uncached
attention recomputation exceeding the compute saved by skipping layers) not a
fundamental limit of the idea. The overall contribution of this work is a
real, modest, well-controlled **interpretability finding** — the signal is
decodable and calibratable but not shown to be causal — accompanied by a
**negative-to-mixed systems result** with a diagnosed, addressable cause. This
should not be read as a systems contribution as currently implemented.

## 2. Introduction

Speculative decoding accelerates autoregressive generation by having a small
drafter model propose several tokens, then verifying all of them against the
target model in a single parallel forward pass, accepting or rejecting each
according to an exact probabilistic rule that preserves the target's output
distribution losslessly. The target's verification step, as normally
implemented, always pays the full cost of every one of its layers for every
proposed token, regardless of how easy or hard that token's acceptance
decision actually is — a trivially predictable continuation (e.g. completing
a common phrase) is verified with exactly as much compute as a genuinely
uncertain one.

This project asks two questions, and its structure should be read as
**two tracks, one conditional on the other, not one linear pipeline**:

- **The primary, interpretability question**: is the target's eventual
  accept/reject verdict for a drafted token already decodable from an
  *intermediate* hidden state, before the forward pass that would compute it
  exactly has finished? And if decodable, is that signal *causally* used by
  the model's own computation, or merely correlated with it?
- **The secondary, conditional, systems question**: *if* the signal is
  decodable, can it be used to skip computation — a gate that pre-empts an
  accept decision using an intermediate-layer readout, skipping the
  remaining layers for that token? This question was only ever worth asking
  contingent on a positive answer to the first, and — as detailed below —
  proceeding to it after a *negative* causal finding was a deliberate,
  explicitly-flagged choice to test "predictive is enough" as a hypothesis
  in its own right, not an assumption that decodability implies usability.

Stating this structure upfront matters because the project's actual results
land asymmetrically across the two tracks: a real interpretability finding,
and a systems result that is negative more often than not. A reader who
expects a systems paper that happens to include some probing analysis will
be misled; a reader who expects an interpretability paper that also
honestly reports what happens when you try to cash the finding out for
speed will get an accurate picture.

## 3. Related work

*(Positioning below reflects the design discussion already had for this
project; general characterizations of prior work are at the level of
well-known method descriptions rather than reproduced numerical claims from
those papers.)*

**Speculative decoding and its exact accept/reject rule.** This project
implements the accept/reject rule of Leviathan & Chen (2023) (and
independently, Chen et al.'s concurrent DeepMind formulation) from scratch:
a drafted token $x$ sampled from the drafter's distribution $q$ is accepted
with probability $\min(1, p(x)/q(x))$ under the target's distribution $p$;
on rejection, the next token is resampled from the residual distribution
$\max(0, p - q)$, renormalized. This construction is what makes speculative
decoding *exactly* lossless with respect to the target's own output
distribution — not an approximation. Any modification to this rule (as
Phase 5's gate is) inherits the burden of showing it preserves — or
honestly reports how it fails to preserve — that same property.

**Self-speculative and early-exit decoding.** A separate family of methods
(architecturally represented by approaches like LayerSkip, SWIFT, and
draft-and-verify-style early-exit methods such as DEL, and hierarchical
variants like HiSpec) accelerate generation by having a single network draft
*with itself* — typically by exiting early through a subset of its own
layers to produce a cheap draft, then verifying with the network's full
depth. These methods are effective, but they are a fundamentally different
design point from what this project targets: the drafter and verifier are
the same weights, so there is no independent drafter distribution $q$ to
speak of, and the "speculative" framing is closer to an early-exit
self-consistency check than to the two-model draft/verify split this
project's brief specified.

**EAGLE** sits at a particularly relevant boundary case. EAGLE's draft head
is trained to propose tokens using the *target's own* hidden state at the
previous position as its input feature, and reuses the target's embedding
matrix and LM head. This is an effective and popular method, but it
violates the decoupling this project's brief required for its own drafter:
a draft head that consumes the target's hidden state and shares its
input/output embedding space is drafting *from* the target, not
independently *of* it — a form of self-speculative decoding under a
different name, even though it is packaged as a separate small network.
Phase 0 of this project flagged this tension explicitly and chose a
genuinely independent, separately pretrained and instruction-tuned drafter
(Qwen2.5-0.5B-Instruct) instead, at the cost of losing whatever extra
proposal-quality EAGLE's target-conditioned drafting would have bought.

**This project's specific novelty** relative to both families above is not
the speculative decoding mechanism itself (implemented exactly, not
modified, through Phase 2) but the question asked of the *target's own*
verification computation: given a properly decoupled, independent drafter
— so that any signal found in the target's hidden states cannot be an
artifact of the drafter secretly being a function of the target — is the
target's forthcoming accept/reject verdict already linearly or near-linearly
decodable partway through its own forward pass? This is an interpretability
question about the target model in isolation, asked in a setting
(speculative decoding) chosen because it provides a well-defined, externally
meaningful ground-truth label (the exact accept/reject rule) for what
"decodable" should mean, rather than an internally-defined proxy task.

## 4. Method

**Models.** Target: Qwen2.5-7B-Instruct (28 decoder layers, hidden dim
3584). Drafter: Qwen2.5-0.5B-Instruct (24 layers, hidden dim 896). Same
tokenizer family (shared vocab, size 151643 real tokens; note the raw
logit tensor is zero-padded differently per model — 152064 vs. 151936 —
so $p$/$q$ must be computed by slicing to the shared real vocabulary size
before any comparison, not over the padded width). Both fully open weights,
no gated-repo friction. Both comfortably fit on a single H200 in bf16.

**Decoupled drafter, not EAGLE.** As discussed in Related Work: the brief
allowed either an EAGLE-style checkpoint or an independent small LM, but
also required strict decoupling (no shared weights or forward pass between
drafter and target). These requirements are in tension specifically for
EAGLE, whose draft head takes the target's own hidden state as input. The
project chose the independent-LM branch: Qwen2.5-0.5B-Instruct is a
separately pretrained and instruction-tuned checkpoint, not derived from
the 7B model's own layers or weights.

**The exact accept/reject rule**, implemented from scratch (no
`model.generate(assistant_model=...)` library shortcut, so every
intermediate quantity is inspectable): the drafter proposes $k=4$ tokens
autoregressively; the target does one forward pass over
(context + $k$ drafted tokens); each drafted token $x_i$ is accepted with
probability $\min(1, p_i(x_i)/q_i(x_i))$, walked sequentially; on the first
rejection, the next token is resampled from the renormalized residual
$\max(0, p - q)$ and the round stops; if all $k$ are accepted, one bonus
token is sampled directly from the target's own next-token distribution.
Temperature is fixed at 1.0 for both models (plain softmax sampling, no
top-k/top-p truncation — truncated sampling would change what distribution
is being compared and would need its own correctness argument). This
implementation was validated by independently recomputing $p$ and $q$ for
every logged token via a completely separate forward-pass code path; all
recomputed values matched the logged ones exactly on a validation set of
generations.

**Training label vs. evaluation label — a deliberate distinction.** The
probes (Phase 3) are trained against the *deterministic* quantity
$\min(1, p(x)/q(x))$, not the stochastic accepted/rejected outcome that
quantity is used to sample — training against the stochastic outcome would
mean training against genuine label noise (the same $p$, $q$ can produce
either outcome depending on the random draw). But AUROC and calibration are
inherently statements about how a predicted probability relates to
*realized* binary outcomes, so evaluation necessarily uses the actual
stochastic `accepted` outcome as ground truth — this is not an
inconsistency, it is the standard way any probabilistic forecast is
validated (a rain-probability forecast is checked against whether it
rained, not against some internal deterministic proxy for rain).

**No incremental KV-cache**, throughout every phase of this project
including the Phase 5 gate: every drafting step and every verification step
recomputes the forward pass over the full sequence so far, rather than
carrying `past_key_values` across rounds. This was a deliberate
correctness-first simplification made at Phase 1 — caching correctly
requires cache truncation/rollback logic on rejection (the drafter's cache
runs ahead of confirmed tokens; the target's cache holds keys/values for
drafted positions that may never be confirmed), which is a real source of
subtle bugs that is hard to hand-verify, and was judged not worth the
complexity at the project's data scale (100k-token generation runs
completed in 35 minutes on an H200 without it). This decision is revisited
as a live limitation in Section 8 and Section 7, since it turns out to be
the direct cause of Phase 5's negative wall-clock results in two of three
domains.

**Domains and data.** Three domains, chosen for open availability and no
licensing friction: code (OpenAI HumanEval, 164 examples), reasoning (GSM8K
test split, 1319 available, subsampled), chat (MT-Bench prompts, 80 unique
examples, first turn only — a documented simplification, not multi-turn
dialogue). All domains are wrapped in the target's chat template uniformly.
100,468 drafted-token records were generated across the three domains
(Phase 2), each logging the hidden state at seven layer depths (stride 4:
layers 4, 8, ..., 28), $p$, $q$, the deterministic label, the realized
accept/reject outcome, and drafter entropy at that step (a token-difficulty
proxy). Acceptance rates and drafter entropy already differentiate the
domains before any probing is done: code 89.0% accept / mean drafter
entropy 0.360; reasoning 87.9% / 0.468; chat 71.2% / 1.307 — chat is
harder by every measure available prior to Phase 3.

## 5. Phase 3 results: decodability and calibration

**AUROC vs. depth.** AUROC rises steadily from layer 4 through layer 24,
then flattens and *slightly declines* by the final layer (28), consistently
for both probe types and across all three domains. Peak overall AUROC:
0.846 (linear, layer 24), 0.859 (MLP, layer 24), versus final-layer (28)
values of 0.838 (linear) and 0.855 (MLP). **This should be stated plainly
as a modest, late-saturating effect, not a dramatic early-saturation
result**: the peak sits four layers before the end of a 28-layer network,
and the AUROC gap between that peak and the final layer is small (0.004–0.008
absolute). The finding supports "decodable somewhat before the full forward
pass completes," not "decodable early in the network" — the peak-number
alone, read without the depth curve behind it, would overstate how early
the signal actually becomes available.

**Domain ordering** is consistent at every layer depth: reasoning >
code > chat in AUROC (at layer 24, MLP: reasoning 0.873, code 0.861, chat
0.803). Chat is the hardest domain to predict from hidden states by a wide
margin, consistent with (though not proven to be caused by) its lower
acceptance rate and substantially higher drafter entropy already observed
in Phase 2 (chat's drafter entropy, 1.307, is roughly 3.6x code's).

**MLP outperforms linear** at every single layer/domain combination
tested, by a fairly consistent 0.01–0.02 AUROC margin — a real but modest
nonlinearity gain. The bulk of the discriminative signal is already
linearly decodable; the MLP is a refinement, not a qualitative jump over
what a linear readout already finds.

**Calibration is where the two probe types diverge sharply**, in a way
AUROC alone does not reveal. Linear probes are poorly calibrated (ECE
approximately 0.25–0.28, roughly flat across all layers and domains)
despite their decent AUROC — reliability curves show the linear probe
systematically *underestimating* acceptance probability (e.g., at a
predicted probability of 0.5, the empirical acceptance rate is often
0.7–0.8). MLP probes are well-calibrated (ECE approximately 0.01–0.03) at
every layer tested. This is a standalone finding independent of the
depth/AUROC result: a linear probe's raw output would need explicit
post-hoc recalibration (Platt scaling or similar) before it could honestly
be used as a confidence threshold for anything; the MLP's output is close
to usable as-is. This is the reason the Phase 5 gate uses the MLP probe,
not the (equally decodable, but uncalibrated) linear one.

## 6. Phase 4 results: causal check

**What was tested.** Whether the Phase 3 probe's layer-24 linear
read-out direction is *causally load-bearing* for the target model's own
verification computation — not merely correlated with, or predictive of,
the eventual accept/reject outcome — via mean-ablation of that direction
(replacing its component of the hidden state with its Phase-3
training-population mean) versus a mechanically identical control ablating
a random direction of the same magnitude.

**Design 1 (diffuse, whole-sequence ablation).** The ablation hook zeroed
the probe's direction at *every* position in the sequence, not just the
position being verified — a real limitation flagged at the time, since this
diffuse intervention could dilute a genuinely localized causal effect.
Result: **negative, and wrong-signed.** Pearson $r$(probe-score,
$\Delta p(x)$) = +0.028 ($p=0.022$) versus the random-direction control's
$r=-0.008$ ($p=0.51$); Spearman $\rho = +0.052$ ($p<0.001$) versus control
$\rho=-0.010$ ($p=0.41$). The causal hypothesis predicts a *negative*
correlation (high-confidence examples should lose real acceptance
probability when the direction is removed); the observed correlation was
weakly positive — the wrong sign. Effect sizes were tiny in both
conditions (mean $|\Delta p(x)|$: 0.0031 real vs. 0.0025 control) and, in
the code domain specifically, the *random* control showed a
larger-magnitude effect ($r=-0.049$, $p=0.021$) than the real probe
direction ($r=+0.038$, $p=0.077$) — the control outperforming the
treatment in one domain, which argues against the effect being real there
at all.

**Design 2 (position-restricted ablation)**, run specifically to address
Design 1's diffuse-intervention caveat: the ablation was restricted to only
the single sequence position being verified for each drafted token (one
forward pass per position rather than one batched pass per round), with
identical seeds/prompts/generator call order so the clean-pass generation
trajectory is bit-for-bit identical to Design 1's, making the two results
directly comparable. Result: **still negative**, though the picture shifted
in an interesting way. Pearson $r = -0.016$ ($p=0.204$, not significant) —
now the hypothesis-predicted sign, but not statistically distinguishable
from zero. Spearman $\rho = -0.033$ ($p=0.007$) *was* significant, and
clearly separated from the control's $\rho=-0.001$ ($p=0.94$). But the
effect size remained negligible in absolute terms (mean $|\Delta p(x)|$:
0.0028 real vs. 0.0023 control, the same order of magnitude as Design 1),
and the code-domain inversion persisted: the random control again showed a
larger, more significant effect ($r=-0.056$, $p=0.009$) than the real probe
direction ($r=-0.019$, $p=0.373$).

**Integrated conclusion.** Ruling out the diffuse-intervention confound
(Design 2) did not reveal a hidden strong effect that Design 1 had merely
diluted — it produced a technically correct-signed but still practically
negligible correlation, with the same domain-level inconsistency (control
beating treatment in code) as before. Combined across both designs: the
probe's layer-24 direction is linearly decodable and predictive of
accept/reject outcomes (Section 5) but is **not shown to be causally
load-bearing** for the target's own verification computation, under two
independent ablation designs with matched controls. This is not proof of
absence of a causal role — transformer representations are known to
exhibit redundant, superposed feature encoding, where ablating one
direction often produces a smaller effect than naively expected because
the same information remains recoverable from elsewhere in the residual
stream, and this project's ablation method (mean-ablation toward a
population mean, not interchange/activation patching between two live
generations) is one specific causal-intervention design among several
possible ones. But as tested, the causal claim that would most cleanly
justify the systems use explored next did not hold up.

## 7. Phase 5 results: verification-skipping gate

**Framing, restated explicitly.** Everything in this section proceeds on
"predictive is enough for a systems shortcut" grounds only, given Section
6's negative causal finding — not because the mechanism by which the gate
might work is understood. The theoretical ceiling on savings was stated
upfront as modest: skipping layers 25–28 covers 4 of 28 decoder layers
(≈14.3% of depth), and — as detailed below — even that ceiling is not
automatically realized, since every SD round still needs at least one real,
full-depth verification regardless of gating (for the bonus token, or to
confirm an actual rejection), and this project's no-KV-cache design means
that real verification, when needed more than once within a round,
recomputes overlapping attention work from scratch each time.

**Gate design.** At each drafted position, the layer-24 hidden state is
standardized with Phase 3's frozen train-split statistics and passed
through the trained MLP probe (layer 24 — the AUROC peak — and MLP
specifically, for the calibration reason established in Section 5, since a
threshold-based gate is only honest if "0.98 predicted probability"
actually means approximately 98% empirical acceptance). If the predicted
probability clears a threshold, the gate commits the token immediately and
skips computation through the remaining four layers for that position; if
not, the exact accept/reject rule runs as normal, on a real (non-skipped)
forward pass. **The gate can only pre-empt an accept, never decide a
rejection** — there is no code path by which the gate's output causes a
rejection; a rejection can only come from the real accept/reject draw,
which by construction only executes when the gate does not fire. The
layer-truncation mechanism itself (swapping the model's decoder-layer list
for a shorter one and reusing the library's own forward code, rather than
reimplementing decoder internals by hand) was verified numerically before
use: the split prefix/suffix computation reproduces a real full-depth
forward pass's logits exactly (max absolute difference 0.00000 across
test prompts, identical argmax).

**Threshold choice.** Two thresholds were tested — 0.98 and 0.995 — the
first picked from a coarse calibration-bin inspection of Phase 3's
saved reliability data (top bin, predicted probability 0.933–1.0, mean
confidence 0.988, empirical acceptance 0.983), the second as a stricter
cutoff tested after the first proved too lossy.

**Losslessness validation — not lossless at either threshold, tested as a
live-generation question, not a static calibration extrapolation.** On 60
held-out prompts (20 each, code/reasoning/chat) with a 128-token cap,
generation was run twice per prompt with identical seeds: once with the
exact ungated rule, once with the gate's decisions driving the trajectory
(while the real accept/reject outcome was also computed alongside, for
comparison, without affecting what was generated). Because generation is
autoregressive, a single gate error early in a completion cascades: every
one of the diverged prompts in this study had exactly one gate mismatch
(gate fired where the real rule would have rejected), and every prompt
with zero mismatches matched the ungated output exactly — a fully
explained, non-noisy relationship.

| | threshold=0.98 | threshold=0.995 |
|---|---|---|
| gate fire rate — overall | 32.5% | 24.5% |
| local mismatch rate — overall (gate wrong \| gated) | 0.41% | 0.18% |
| **exact-output match rate — overall** | **85%** | **95%** |
| exact-output match rate — code / reasoning / chat | 75% / 85% / 95% | 95% / 90% / 100% |

Raising the threshold bought a disproportionate reliability gain relative
to how much gating opportunity it cost (fire rate fell ~25% relative,
divergence fell ~67% relative) — evidence the gate design is not doomed by
construction — but **threshold=0.995 is still not lossless**: 5% of
completions overall, and 10% of reasoning completions specifically (2 of
20), come out different from what the exact algorithm would have produced.
This is reported as a real, nonzero cost throughout the rest of this
section, not rounded down to zero.

**Wall-clock benchmarking, threshold=0.995 only** (0.98 was not
benchmarked; its divergence rate already ruled it out). Using the real
layer-skipping code path — not the validation path above, which
deliberately paid full compute — on the same 60 prompts, same seeds. Gate
fire rate this run (24.6%) matched the losslessness run (24.5%) within
noise, confirming no discrepancy between the two runs' gating behavior.
Every speedup number below must be read paired with the divergence rate
established above — it is not repeated at every mention below only for
brevity, not because it stops applying.

| domain | mean speedup | median speedup | gate fire rate |
|---|---|---|---|
| **overall** | 1.09x | **1.02x** | 24.6% |
| code | 1.34x | 1.08x | 37.0% |
| reasoning | **0.94x (slower)** | **0.92x (slower)** | 27.8% |
| chat | 0.96x (slower) | 0.97x (slower) | 10.1% |

Probe overhead was measured separately (not interleaved with the timing
loop, to avoid contaminating it with extra synchronization): 82.3
microseconds per call, 6740 total calls across the whole benchmark, summing
to an estimated 0.55 seconds against 171.1 seconds of total gated
wall-clock — about 0.3%. **The probe itself does not explain the
slowdowns.**

**Why reasoning was slower and code was faster: a supplementary analysis.**
This was not established by the numbers above and required one additional,
narrowly-scoped rerun (`scripts/11_gate_round_analysis.py`, same 60
prompts/seeds/threshold, no new mechanism code — reusing the existing
validation-mode verify step, but preserving per-round boundaries instead
of only aggregate counts) to test three candidate explanations without
guessing among them: (1) reasoning completions simply being longer, making
every uncached recompute costlier in absolute terms; (2) gated positions
firing later in the sequence, where recompute would have been more
expensive; (3) reasoning producing a costlier pattern of real-verification
fallbacks per round.

*(Note: an initial version of this analysis inferred round boundaries
after the fact from gaps in token position, which — a real bug caught
before reporting, not before running — silently merged consecutive rounds
whenever a round ended in an ordinary rejection rather than a full
acceptance, since only the latter leaves a detectable position gap. It was
corrected by tracking round boundaries directly during generation before
any numbers below were finalized.)*

**Mechanism 1 (length) is ruled out.** Generated completion length is
essentially identical across domains: code 129.9 mean tokens, reasoning
129.2, chat 129.45 (all near the 128-token cap; none of the domains ended
substantially early on EOS in this sample). Length differences cannot
explain the domain asymmetry.

**Mechanisms 2 and 3 are both partially supported, and appear to compound
rather than either one dominating alone:**

| | code | reasoning | chat |
|---|---|---|---|
| gate-fire position: early / mid / late thirds of completion | 11.5% / 35.7% / 52.8% | 16.3% / 42.8% / 40.8% | 24.1% / 39.8% / 36.1% |
| mean real-verification events per round | 2.05 (median 2) | 2.46 (median 3) | 2.13 (median 2) |
| rounds with ≥1 real-verification event | 84.5% | 95.6% | 98.1% |
| mean rounds per prompt | 34.2 | 31.85 | 49.6 |

Code's gate fires disproportionately in the *later* third of a completion
(52.8% of its firing events, versus 40.8% for reasoning and 36.1% for
chat) — meaning code disproportionately skips the positions where a real
verification call would have been most expensive (suffix cost scales with
how far into the sequence the position sits), banking more savings per
averted event than reasoning does. Separately, reasoning triggers a real
(uncached, no-savings) verification event in 95.6% of its rounds, more
often than code's 84.5%, and needs somewhat more such events per round on
average when it does (2.46 vs. 2.05) — paying the redundant-recomputation
cost more consistently. Chat is a distinct case from both: its gate fires
so rarely (10.1% overall) that its behavior stays close to the ungated
baseline throughout (98.1% of rounds have a real event, nearly identical
to what full verification would look like anyway), which is consistent
with its comparatively small measured slowdown despite being the domain
that benefits least from the mechanism.

**This should be read as: the data supports a two-factor, compounding
explanation (where gating concentrates in the sequence, and how often
per-round real verification recurs), not a single dominant cause, and does
not support the length-based explanation at all.** Neither factor alone
fully accounts for the asymmetry; both point in the same direction (favoring
code, disfavoring reasoning) and are of comparable, modest magnitude.

**Plain verdict.** At threshold=0.995, this gate produces a **net-negative
or negligible result in two of three domains** — reasoning is both
measurably slower and still the domain with the highest residual
divergence rate (10%), the worst combination in the results; chat is
measurably slower though fully lossless in this sample. Code is the only
domain with a real speedup (1.34x mean / 1.08x median — the median is the
more trustworthy number given single-run-per-prompt methodology on n=20;
see Section 8). The pooled overall figures (1.09x mean / 1.02x median) are
arithmetically dominated by code and should not be read as evidence of a
general win. **This does not support the gate as a systems contribution as
currently designed.** The diagnosed cause — no-KV-cache recomputation cost
exceeding the four-layer savings in two of three domains — is a specific,
addressable limitation of this implementation, not a demonstration that
the underlying idea cannot work; a KV-cached suffix path is the natural
next experiment, explicitly not attempted here (Section 8).

## 8. Discussion and limitations

**Single model pair.** Every result in this report — decodability,
calibration, the causal-ablation finding, and the gate's behavior — comes
from one target/drafter pair (Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct).
Whether the AUROC-vs-depth curve's shape, the specific layer at which it
peaks, the causal-ablation null result, or the domain-dependent Phase 5
speedup pattern would hold for a different model family, a different
size ratio between target and drafter, or a different tokenizer/vocabulary
setup is untested. Nothing here should be read as a claim about
transformers in general.

**Small sample sizes in several places.** The causal check (Phase 4/4b)
used 60 prompts total (20 per domain) for its live-generation runs; the
chat domain throughout the project is limited to 80 unique underlying
MT-Bench questions (cycled across multiple generation passes in Phase 2,
with the pass-suffix collapse handled explicitly in the train/val/test
split to avoid leakage). Phase 5's wall-clock benchmark used **one timed
run per prompt per condition** — no repeated trials to establish a
per-prompt noise estimate, only cross-prompt spread across 60 distinct
prompts. The gap between code's 1.34x mean and 1.08x median speedup is
itself evidence of a heavy-tailed, possibly noisy distribution at this
sample size; the median is reported as the more trustworthy figure for
exactly this reason, not because the mean was inconvenient.

**The speedup numbers are relative to this project's own uncached
reference implementation, not a production speculative-decoding stack.**
Every wall-clock number in Section 7 compares the gated path against this
project's own ungated baseline, which — per the Method section — never
implements KV-caching, by a Phase 1 design decision made for
correctness-first simplicity at small scale. A production SD system with
proper cache management would have a different (almost certainly lower)
absolute per-round cost than this project's baseline, which changes the
denominator the gate's savings are measured against. The Phase 5 verdict
should be read as "this specific gate mechanism, layered on this specific
uncached baseline, is net-negative in two of three domains" — not as a
general statement about verification-skipping gates layered on
production-grade SD infrastructure.

**Whether a KV-cached implementation would change the Phase 5 verdict is
an open question, explicitly not attempted here.** Section 7's diagnosed
mechanism (redundant attention recomputation when a round needs more than
one real-verification event) is precisely the kind of inefficiency a
suffix-layer KV-cache would address. Given the project's decision to keep
Phase 1 uncached for a documented reason, and Phase 5's diagnosis showing
that decision has a real, measured cost for this particular downstream
use, revisiting it is the most concrete, well-motivated next step — stated
here as future work, not undertaken as part of this report.

**The causal-ablation null result is not proof of no causal role.**
Section 6's conclusion is explicitly a "not shown to be causal," not a
"shown to be non-causal." Only one causal-intervention family
(mean-ablation toward a population mean, in two variants) was tested;
interchange or activation-patching designs between two live generations
are a different, in some ways stronger, method not attempted here. The
redundant/superposed-representation interpretation offered in Section 6 is
plausible given known properties of transformer representations, but it is
an interpretation consistent with the null result, not something this
project's experiments independently established.

## 9. Conclusion

This project set out to test whether a target model's verification verdict
in speculative decoding is decodable from its own intermediate hidden
states, and — conditionally — whether that decodability could be turned
into a systems speedup. The honest scope of what was found is: a real,
modest, well-controlled interpretability result (a layer-24 MLP probe
predicts the exact accept/reject label with AUROC 0.86, well-calibrated,
peaking a few layers before the network's end rather than dramatically
early), paired with a negative causal-ablation result under two
independent designs (the same direction that predicts the outcome is not
shown to be what the model itself causally relies on to compute it), and a
systems experiment built on the "predictive is enough" fallback that
returned a mostly negative, domain-dependent result with a specific,
diagnosed, addressable cause rather than a fundamental one. Every one of
these findings is stated at the strength the evidence supports: the
decodability finding is real but should not be overstated as dramatic
early saturation; the causal result is a well-controlled negative, not
proof of absence; the systems result is a measured net loss in two of
three domains, with a modest and only medium-confidence gain in the
third. None of this should be rounded up into a systems contribution. It
is an interpretability finding, honestly reported, with a documented and
mostly unsuccessful attempt to cash it out for speed.
