# Progress Log — Hidden-State Acceptance Probes for Speculative Decoding

## Status: Causal-check thread closed (Phase 4 + 4b, both negative) -- awaiting user decision on whether/how to proceed to Phase 5

## Compute environment
- Remote node: H200 node (`103.180.163.218`), user `anish`, home `/mnt/data/anish`.
  Chosen over the alternate RTX 6000 Pro node (`103.13.114.78`) because it had
  less contention from other tenants (6-7 of 8 GPUs idle vs 4 of 8) and pip
  was already on PATH. Both nodes are **shared, multi-tenant, company-owned**
  hardware — no `CUDA_VISIBLE_DEVICES` isolation is enforced by the host, so
  every script in this repo must explicitly pin to one free GPU index rather
  than relying on device 0 / defaults.
- No billing/time-reservation constraint (company-owned hardware), so no
  pressure to batch work into long single sessions.
- Local dev machine (this repo's canonical location) is a MacBook Air M4,
  16GB unified memory, no CUDA — used for editing/reviewing only. All
  model loading and training happens on the remote node via SSH; code is
  synced with rsync.

## Design decisions (flagged)

### 1. Drafter: independent small LM, not EAGLE
The brief allowed either "EAGLE if a checkpoint is readily available" or "a
small independent LM," but also required strict decoupling: the drafter must
not share weights or a forward pass with the target. These two are in tension
for EAGLE specifically — EAGLE's draft head reuses the target's embedding
matrix and LM head, and takes the **target's own hidden state** as its input
to propose the next draft token. That is weight-sharing and forward-pass
coupling by construction, i.e. it *is* a form of self-speculative decoding
wearing a different name.

**Decision:** use a genuinely independent small LM as the drafter instead.
Flagged to user at the Phase 0 checkpoint; awaiting confirmation or override.

### 2. Model pair: Qwen2.5-7B-Instruct (target) + Qwen2.5-0.5B-Instruct (drafter)
- Same tokenizer family → vocab alignment for the accept/reject rule is exact,
  no vocab-remapping logic needed between drafter and target.
- Both fully open on Hugging Face (Apache 2.0, no gated-repo license click),
  unlike Llama-3, which simplifies download on a fresh node.
- 0.5B model is a real, separately pretrained+instruction-tuned checkpoint —
  not derived from the 7B's own layers or weights.
- Both models comfortably fit on a single H200 (144GB) in bf16 with huge
  headroom for activation logging.

## Done
- [x] Diagnosed local machine has no CUDA GPU; located remote compute (two
      candidate nodes provided by user), chose H200 node.
- [x] Verified SSH connectivity, GPU availability, disk space, internet
      access from the remote node.
- [x] Local project scaffolding created: `data/`, `probes/`, `analysis/`,
      `configs/`, `scripts/`, `PROGRESS.md`, git repo initialized.
- [x] Remote Python env: venv at `/mnt/data/anish/hidden-state-probes/.venv`
      on the H200 node, with torch 2.13.0+cu130, transformers 5.14.1.
      Pinned to GPU index 1 (confirmed idle via `nvidia-smi` before use) via
      `CUDA_VISIBLE_DEVICES` — necessary since the box is shared and has no
      host-level GPU isolation between tenants.
- [x] `scripts/00_check_env.py` written and run remotely: loads both models,
      runs one basic (non-SD) forward pass each, checks tokenizer vocab
      compatibility. **Result: PASSED.**
      - Target Qwen2.5-7B-Instruct: 28 hidden layers, hidden dim 3584,
        tokenizer vocab_size 151643. Greedy decode of "The capital of France
        is" -> " Paris". ~15.3GB GPU memory after load (bf16).
      - Drafter Qwen2.5-0.5B-Instruct: 24 hidden layers, hidden dim 896,
        same tokenizer vocab_size 151643, identical token->id mapping as
        target. Same prompt -> " Paris". ~1.0GB GPU memory after load.
      - Note for Phase 1: raw `logits` last-dim is padded differently per
        model (152064 for target, 151936 for drafter) versus the shared
        real vocab_size of 151643. p(x)/q(x) computation must slice/softmax
        over `[:vocab_size]` only, not the padded logit width, or the two
        distributions won't be comparable.

## Phase 1: instrumented SD loop

### Design decisions (flagged)

**No incremental KV cache across rounds.** Every drafting step and every
verification step recomputes the forward pass over the full sequence so far,
rather than carrying `past_key_values` across rounds. Real deployments of SD
use KV caching for throughput, but caching correctly requires cache
truncation/rollback logic on rejection (drafter cache is "ahead" by the
rejected tokens; target cache holds keys/values for drafted positions that
never get confirmed). That bookkeeping is a real source of subtle bugs and
is hard to hand-verify. Chose correctness-first simplicity for Phase 1/2 at
the ~100k-token small-scale stage; full recompute is computationally fine at
this scale on an H200. Flagging that this may need revisiting for
throughput if Phase 2 turns out too slow, but not building the caching
complexity preemptively.

**Layer tap point:** HF's `output_hidden_states=True`, i.e. the residual
stream immediately after each decoder block (standard, unambiguous
convention; `hidden_states[0]` is the embedding output, `hidden_states[i]`
is the output of decoder block `i`).

**Layer stride = 4** (target has 28 layers -> logs layers
`[4, 8, 12, 16, 20, 24, 28]`, 7 points). Since `output_hidden_states=True`
computes all layers regardless (no compute saved by a stride, this is a
storage/IO-only choice), stride 4 was picked as a balance: enough
resolution for an AUROC-vs-depth curve, without logging all 28 layers,
which would not stay budget-conscious once Phase 2 scales beyond 100k
tokens. See `configs/sd_config.yaml` for the exact rationale inline.

**Position indexing for a drafted token.** For a round with existing
context length `n` and `k` drafted tokens fed to the target in one parallel
forward pass over `[context, x_1..x_k]`: drafted token `x_{i+1}` (1-indexed
per the paper) is verified using the target's output at 0-indexed input
position `n - 1 + i`. This is exactly the same conditioning point the
drafter used to sample `x_{i+1}`, and it's the intermediate-hidden-state
tap the probe hypothesis is about ("decodable before the forward pass
finishes" = the layer-k hidden state at that position, before the residual
stream has passed through the remaining `28-k` blocks). Verified against a
completely independent recomputation, see below.

**Sampling temperature = 1.0**, shared between drafter (for proposing +
computing q) and target (for computing p). Plain softmax sampling, no
top-k/top-p truncation, to keep the accept-reject rule exactly as specified
(temperature-truncated sampling changes what distribution is being
compared and would need its own correctness argument).

**Label vs. stochastic outcome.** `min(1, p(x)/q(x))` is logged for every
one of the k drafted tokens in a round, regardless of where the sequential
accept/reject walk actually stops — it's a pure function of p and q, both
already computed by the one parallel target forward pass, so there's no
need to discard "would-have-been-rejected-anyway" tokens as probe examples.
The actual stochastic `accepted` bool is also logged, but only for
bookkeeping/sanity checks, never as a training target.

**Token-difficulty proxy**: drafter entropy at each drafting step is logged
now (cheap, already computed in the drafting loop) even though it's not
consumed until Phase 2/3 analysis.

### Implementation
- `speculative/sd_loop.py`: `draft_tokens()` (drafting phase),
  `verify_and_step()` (one target forward pass + accept/reject/resample
  walk + logging), `run_speculative_decoding()` (drives rounds to
  `max_new_tokens`). Small, single-purpose functions per file, no library
  SD shortcuts used anywhere.
- `configs/sd_config.yaml`: `k=4`, `temperature=1.0`, `layer_stride=4`,
  `seed=0`.

### Validation (`scripts/01_validate_sd.py`)
Ran the loop on 3 short prompts (factual, code, arithmetic) and, for every
drafted token, independently recomputed q(x) and p(x) via **separate fresh
forward passes** (no reuse of any loop-internal state/tensors) — different
code path, same models, to catch indexing/slicing bugs rather than
rubber-stamp the same computation twice.

**Result: 7/7 records matched (0 mismatches)**, e.g.:
```
prompt: 'The capital of France is'
   tok  q(logged)  q(indep)  p(logged)  p(indep)   label  accept
' Paris'    0.2719    0.2719     0.4994    0.4994  1.0000    True
 '.\n\n'    0.0692    0.0692     0.0176    0.0176  0.2543   False
```
Full output in the checkpoint report to user.

A smoke-test full generation (40 tokens, prompt "The capital of France is")
produced a 59.46% acceptance rate over 37 drafted-token records. The
generated continuation wanders off-topic after the correct initial answer
("...is Paris. ..." then unrelated text) — expected at temperature=1.0 with
no repetition penalty on a short factual prompt with nothing more
determinate to say, not a bug signal (the correctness check above is what
matters here, not sample quality).

## Phase 2: data generation

### Design decisions (flagged)

**Datasets**: `openai/openai_humaneval` (code, 164 examples, `prompt` field),
`openai/gsm8k` "main" split=test (reasoning, 1319 examples, `question`
field), `HuggingFaceH4/mt_bench_prompts` (chat, 80 examples). All fully
open, no gating.

**Chat domain uses first turn only.** MT-Bench prompts are multi-turn (a
`prompt` field containing a list of turns); using only turn 1 as a
single-shot chat prompt is a simplification to avoid building multi-turn
conversation state into Phase 2. Flagging this: the "general chat" domain
data is single-turn-instruction-following, not multi-turn dialogue.

**All domains wrapped in the target's chat template** (`tokenizer.apply_chat_template`,
`add_generation_prompt=True`), including code and reasoning -- i.e. the
model is prompted as an instruct assistant uniformly across domains rather
than doing raw code-completion-style continuation for HumanEval. Simpler,
consistent pipeline; means code completions may include prose/markdown
fences around the code, which is fine since we're studying accept/reject
dynamics, not code correctness.

**EOS-aware stopping added** to `run_speculative_decoding` (checked against
Qwen's actual turn-end token `<|im_end|>`, id 151645, not the tokenizer's
base `eos_token_id` attribute -- confirmed these coincide for this model).
Generation stops once EOS is among the *confirmed* tokens for a round
(accepted drafts + resample/bonus), not on a merely-proposed-but-rejected
EOS. Without this, every generation would run to the 256-token cap
regardless of natural response length, and post-EOS "continuation" tokens
would be degenerate probe examples in a domain-imbalanced way (chat
responses are typically shorter than a code function body).

**Per-domain token budget cycles through the prompt list repeatedly**
(reshuffled, new seeds each pass) if the budget isn't met in one pass over
the domain's prompts. This matters most for chat, which has only 80 unique
underlying prompts -- see caveat below.

**Caveat for Phase 3 (train/val/test split by prompt) — flagging now so it
isn't a bug later:** the `prompt_id` field logged per record is
`f"{domain}_{idx}_pass{pass_idx}"`, so a chat example revisited on pass 2
gets a *different* `prompt_id` string than on pass 1, even though it's the
same underlying MT-Bench question (just resampled with a different seed).
Splitting naively by the literal `prompt_id` string would leak
near-duplicate content (same question, different completion) across
train/val/test for the chat domain. **Phase 3 must split chat by the
underlying question index** (`idx`, stripped of the pass suffix), not by
the full `prompt_id` string. Code and reasoning domains are unaffected
(each used <= 1 pass, so `prompt_id` already uniquely identifies a source
prompt there).

### Results

100,468 drafted-token records total, 35.5 min wall clock, 5.04GB on disk
(`data/phase2_small/`, fp16 hidden states at 7 layers x 3584 dims/record).

| domain | records | unique source prompts used | passes | accept rate | label mean | label std | frac label==1.0 | frac label<0.1 | drafter entropy (mean) |
|---|---|---|---|---|---|---|---|---|---|
| code (HumanEval) | 33,526 | 155 / 164 | 1 | 89.0% | 0.889 | 0.286 | 81.8% | 7.1% | 0.360 |
| reasoning (GSM8K) | 33,396 | 154 (of 1319 avail.) | 1 | 87.9% | 0.879 | 0.292 | 77.2% | 7.4% | 0.468 |
| chat (MT-Bench) | 33,546 | 145 prompt_ids / 80 underlying | 2 | 71.2% | 0.711 | 0.407 | 57.4% | 19.0% | 1.307 |

Label distribution is bimodal in all domains (heavy mass at label=1.0 --
deterministic accept -- plus a smaller low-probability tail), which is
expected for an accept-probability target. Chat is clearly the "harder"
domain by every measure here: lowest acceptance rate, lowest mean label,
highest label variance, highest drafter entropy (~3.6x code's) -- consistent
with the project hypothesis that domains differ in token difficulty, and a
reasonable signal that chat will be the domain where probe AUROC-vs-depth
is most informative to look at in Phase 3.

### Implementation
- `speculative/datasets.py`: domain prompt loaders.
- `speculative/storage.py`: `ShardWriter` (one row per drafted token,
  hidden states as `[num_layers, hidden_dim]` fp16, `prompt_id` per row).
- `scripts/02_generate_data.py`: orchestrates generation across domains
  with budget + pass-cycling; writes `manifest.json` with per-domain stats.
- Validated via a 200-token/domain dry run first (correct label range,
  correct hidden_states shape, ~52KB/record on disk) before launching the
  full run.

## Next
- Phase 3: linear + MLP probes per logged layer, predicting the `label`
  field from `hidden_states`. Train/val/test split **by prompt** (with the
  chat-domain pass-suffix caveat above handled). Report AUROC + calibration
  (ECE, reliability diagrams) per layer, per domain, for both probe types.

## Phase 3: probe training

### Design decisions (flagged)

**Training target vs. evaluation target are deliberately different.**
Probes are trained with soft-label `BCEWithLogitsLoss` against the
deterministic `min(1,p/q)` label (as required -- avoids the coin-flip
label-noise problem). But AUROC and calibration (ECE, reliability
diagrams) are inherently about how well a predicted probability matches
*realized* binary outcomes -- there's no other honest way to define them.
So evaluation uses the actual stochastic `accepted` outcome as ground
truth. This is the standard way any probabilistic forecast gets validated
(e.g. a weather model's rain probability is checked against whether it
actually rained, not against some internal deterministic proxy). Not an
inconsistency -- training avoids noisy labels, evaluation necessarily uses
realized outcomes because that's what AUROC/calibration mean.

**Pooled-domain probes, not per-domain.** One probe per (layer, probe
type) trained on data pooled across all three domains, then evaluated with
results sliced by domain. Chosen over training separate per-domain probes
because the project's hypothesis is about whether the accept/reject signal
is decodable at all (a general claim), and pooling directly tests whether
that signal is domain-general vs. domain-specific by looking at the
per-domain slice of a domain-general probe's performance. Flagging as a
default -- happy to also train per-domain probes as a follow-up comparison
if useful.

**Features standardized per layer** (zero mean, unit variance, fit on
train split only, applied to val/test) before both probe types -- residual
stream activation norms are known to grow with depth in transformers, so
this avoids layer-to-layer comparisons being confounded by raw scale
rather than genuine signal.

**Split by prompt** using `probes/splits.py`, handling the chat-domain
pass-suffix caveat flagged in Phase 2 (`chat_12_pass0` and `chat_12_pass1`
collapse to the same split key so they can't land in different splits).
70/15/15 train/val/test, stratified by domain. 389 unique prompts total
(155 code, 154 reasoning, 80 chat -- chat's low count is the known
consequence of only 80 unique MT-Bench questions, cycled across passes).

**MLP architecture**: single hidden layer, 256 units, ReLU, dropout 0.1 --
kept deliberately small relative to ~70k training rows, meant as a
nonlinearity check against the linear probe, not a capacity race.

**Compute**: ran entirely on CPU on the H200 node (192 cores, 2TB RAM
available) rather than GPU. Probes are tiny relative to the stored
activations (no forward pass through the 7B/0.5B models needed at all,
just the already-extracted hidden states), and the node's GPUs were
occupied by other tenants' jobs at the time -- CPU was both sufficient and
avoided touching shared GPU resources unnecessarily. Full run (14 probes:
7 layers x {linear, mlp}) took ~4.5 minutes.

### Results

`analysis/phase3/results.json`, `reliability.json`, plots in
`analysis/phase3/plots/`.

**AUROC vs. depth** (`auroc_vs_depth.png`): AUROC rises steadily from
layer 4 through layer 24, then **flattens and slightly declines by layer
28 (the final layer)** -- for both probe types and consistently across
domains. Peak overall AUROC: linear 0.846 (layer 24), MLP 0.859 (layer
24), vs. final-layer (28) values of 0.838 (linear) and 0.855 (MLP). This
is a real, if modest, instance of the project's core hypothesis: the
accept/reject signal saturates *before* the final layer rather than
requiring the full forward pass.

Domain ordering is consistent at every layer: **reasoning > code > chat**
in AUROC (e.g. at layer 24, MLP: reasoning 0.873, code 0.861, chat 0.803).
Chat is the hardest domain to predict from hidden states by a wide margin,
consistent with its lower acceptance rate and higher drafter entropy from
Phase 2.

MLP outperforms linear at every single layer/domain combination, by a
fairly consistent ~0.01-0.02 AUROC margin -- a real but modest nonlinearity
gain. The bulk of the discriminative signal is already linearly decodable;
the MLP is a refinement, not a qualitative jump.

**Calibration vs. depth** (`ece_vs_depth.png`): a much starker difference
than AUROC. Linear probes are **poorly calibrated** (ECE ~0.25-0.28,
roughly flat across all layers/domains) despite having decent AUROC.
MLP probes are **well-calibrated** (ECE ~0.01-0.03) at every layer.
Reliability diagrams (`reliability_diagrams.png`, layers 4/16/28) confirm
this visually: linear-probe curves sit well above the diagonal across most
of the probability range (the probe *underestimates* acceptance
probability -- e.g. at predicted prob. 0.5, empirical accept rate is often
~0.7-0.8), while MLP curves track the diagonal closely at every depth
shown. **Practical implication for Phase 5** (if we get there): a linear
probe's raw output would need explicit recalibration (e.g. Platt
scaling/temperature scaling) before being used as a "definitely-accept"
confidence threshold; the MLP's output is closer to usable as-is.

### Implementation
- `probes/data.py`: shard loader, `split_key` construction (pass-suffix
  stripped).
- `probes/splits.py`: prompt-level, domain-stratified split.
- `probes/models.py`: `LinearProbe`, `MLPProbe`.
- `probes/train.py`: soft-BCE training loop, early-stopped on val AUROC
  (evaluated against `accepted`, matching the eval/train-target split
  above).
- `probes/metrics.py`: AUROC, ECE, reliability-diagram bin stats.
- `scripts/03_train_probes.py`: orchestrates all (layer x probe type)
  combinations, saves `results.json` (per layer/probe/domain metrics) +
  `reliability.json` (calibration bin data) + probe weights.
- `scripts/04_plot_results.py`: generates the three plots above from
  `results.json`/`reliability.json`.

## Next
- Phase 4: causal check (activation patching / ablation) on the
  best-performing layer/probe (layer 24, MLP looks like the natural
  candidate given the results above) -- test whether the decoded signal is
  causally used by the target's own verification outcome, or merely
  correlated with it.

## Phase 4: causal check (activation ablation)

### Design

**Method: targeted mean-ablation via forward hook**, not interchange
patching. At layer 24 (the Phase 3 AUROC peak), take the trained linear
probe's weight direction in raw hidden-state space (`w / sigma`,
normalized to unit length -- accounts for the per-dimension standardization
the probe was trained on), and replace exactly that component of the
hidden state with its training-population mean (`mu . direction`, reusing
statistics already saved from Phase 3, no extra calibration pass needed).
Resume the real forward pass through layers 25-28 + LM head to get the
actual re-computed p(x) for the drafted token. Implemented via a PyTorch
forward hook on `model.model.layers[23]` (verified experimentally that its
output is a plain tensor in this transformers version, and that a no-op
hook exactly reproduces unhooked logits, before writing the ablation math).

**Control: identical mechanics on a random unit direction** (fixed seed),
using the same mean-ablation formula (`mu` dotted with any direction gives
that direction's population-mean value, so the control isn't a different,
weaker method -- it's the exact same intervention aimed at an unrelated
direction). This isolates whether the *specific* probe direction matters
more than an arbitrary perturbation of the same "size."

**Clean pass drives generation; ablated passes are measurement-only.**
Each round runs the target forward 3x on the identical (context,
draft_ids): clean, probe-direction-ablated, random-direction-ablated. Only
the clean pass's accept/reject walk determines what token actually gets
appended to the conversation -- the other two never affect generation,
only get logged for analysis. This keeps generated text coherent and
avoids the ablated conditions cascading into each other's context.

**Fresh prompt sample**, not a replay of Phase 2/3's exact test split (Phase
2 didn't log full token sequences, only isolated per-position data, so
exact replay wasn't possible without re-deriving generation seeds). 20
prompts/domain, capped at 128 new tokens/prompt, same domain loaders as
Phase 2. Since the frozen Phase 3 probe is what's being tested (not
re-trained here), using a fresh sample rather than the literal held-out
test set doesn't compromise the causal question being asked.

**Metric**: Pearson/Spearman correlation between the probe's own
pre-ablation prediction (`probe_score`, computed from the clean-pass
hidden state) and the ablation-induced shift `delta_p = p_ablated -
p_clean`. If the direction is causally load-bearing the way probing
usually hopes, high-confidence examples should lose real acceptance
probability when the direction is removed, and low-confidence examples
should gain some back (regression toward the population mean) -- a
negative correlation. A random direction should show a much weaker (or
absent) version of this.

### Results (negative)

6,736 drafted-token records (60 prompts across 3 domains). Full numbers in
`analysis/phase4/causal_analysis.json`, scatter plot in
`analysis/phase4/causal_scatter.png`.

| | probe direction | random direction (control) |
|---|---|---|
| Pearson r | +0.028 (p=0.022) | -0.008 (p=0.51) |
| Spearman rho | +0.052 (p<0.001) | -0.010 (p=0.41) |
| mean \|delta p(x)\| | 0.0031 | 0.0025 |

**This does not support the causal hypothesis as tested.** Three separate
reasons, stated plainly rather than softened:
1. **Wrong sign.** The predicted mechanism implies a negative correlation;
   the observed one is weakly positive.
2. **Barely distinguishable from the random control.** Both effect sizes
   are tiny (r ~0.01-0.03); the probe-direction correlation is only
   statistically significant because n=6736 is large, not because the
   effect is practically meaningful. Mean |delta p(x)| for the real
   ablation (0.0031) is only marginally larger than the control (0.0025).
3. **Per-domain breakdown makes it worse for the hypothesis, not better**:
   in the code domain, the *random* control shows a larger-magnitude
   effect (r=-0.049, p=0.021) than the real probe direction (r=+0.038,
   p=0.077).

The scatter plot shows both conditions as visually flat, near-zero-slope
clouds -- not "probe direction shows a real trend, random direction is
flat," which is what the causal hypothesis would predict.

**Caveat on the experiment itself (a real limitation, not an excuse):**
the hook ablates the direction at *every* position in the sequence, not
just the position being verified. This is a diffuse intervention across
the whole context representation, which could dilute a genuine localized
causal effect. A cleaner follow-up would restrict the ablation to only the
specific verification position. Flagging this as the natural next
experiment if the user wants to push on this further, rather than treating
the current result as a fully conclusive disproof.

**Interpretation**: Phase 3 showed the signal is *decodable* (good AUROC,
improving with depth, saturating before the final layer). Phase 4's
ablation of the specific linear direction the probe reads from does not
show that direction is *causally necessary* for the target's actual
verification computation, at least not via this whole-sequence mean-
ablation method. This is consistent with (though doesn't prove) a
"correlated but not the mechanism" story -- e.g. the probe's direction may
track features (verbosity, syntactic predictability, etc.) that correlate
with acceptance probability without being the specific computational
pathway the model's own softmax relies on, especially given transformers'
known tendency toward redundant/superposed feature encoding, where
ablating one direction often has a smaller effect than expected because
the same information is recoverable from elsewhere in the residual
stream.

### Implementation
- `probes/ablation.py`: direction extraction from a saved probe checkpoint,
  random-direction generation, `AblationHook` (toggleable, reusable across
  clean/ablated passes on the same layer).
- `probes/causal_verify.py`: Phase-4-specific verify step (3x forward
  passes per round, clean pass drives generation).
- `scripts/05_causal_check.py`: orchestrates generation + logging.
- `scripts/06_analyze_causal.py`: correlation analysis + scatter plot.

## Phase 4 follow-up: position-restricted ablation

Re-ran the exact same causal check as Phase 4, with one change: the ablation
hook (`probes/ablation.py`'s `AblationHook`, now takes an optional
`target_positions` list) restricts the mean-ablation to only the single
sequence position being verified for each drafted token, instead of every
position in the sequence at once. Addresses the "diffuse intervention"
caveat flagged at the end of Phase 4. Same probe checkpoint (layer 24,
linear), same random-direction control, same prompts/seeds/generator
call order as the original run (`scripts/07_causal_check_positional.py`,
new `causal_verify_and_step_positional` in `probes/causal_verify.py`) --
clean-pass generation trajectory is bit-for-bit identical to Phase 4's, so
the two result sets are directly comparable. Costs ~3x the forward passes
per record (a separate ablated pass per drafted position rather than one
batched pass per round: `1 + 2k` passes per round of `k=4` vs. 3 before).

Run remotely on the H200 node, GPU 4. `analysis/phase4_positional/`.

### Results (still negative)

Same 6,736 records (60 prompts, same trajectory as Phase 4).

| | probe direction | random direction (control) |
|---|---|---|
| Pearson r | -0.0155 (p=0.204) | -0.0129 (p=0.290) |
| Spearman rho | -0.0328 (p=0.007) | -0.0009 (p=0.939) |
| mean \|delta p(x)\| | 0.00277 | 0.00228 |

The Pearson sign now matches the causal hypothesis's predicted direction
(negative, unlike Phase 4's wrong-signed +0.028), and Spearman rho is
negative and statistically significant this time (p=0.007, vs. random's
p=0.94) -- so restricting the ablation to the verification position did
change the picture slightly, in the hypothesized direction, not just noise
in a random direction. But this doesn't upgrade the finding to a positive
result:
1. **Effect size is still negligible** -- rho=-0.033 explains essentially
   none of the variance; mean |delta p(x)| for the real direction (0.00277)
   is barely above the random control (0.00228), same order of magnitude
   as Phase 4's diffuse version.
2. **Pearson r is not significant** (p=0.204) -- only Spearman clears
   p<0.05, and by a modest margin, not a decisive break from the control.
3. **Per-domain breakdown still undercuts the story**: in the code domain
   the random control again shows a larger-magnitude, more significant
   effect (r=-0.056, p=0.009) than the real probe direction (r=-0.019,
   p=0.373) -- the same inversion Phase 4 found, unresolved by
   position-restriction. Reasoning domain even flips sign for the random
   control (r=+0.012) vs. probe (r=-0.013), both non-significant.
4. **Scatter plots are visually indistinguishable** between the two
   conditions (`analysis/phase4_positional/causal_scatter.png`) -- both
   are flat, near-zero-slope clouds with the same spread, not "probe
   direction shows a real trend, random direction is flat."

**Interpretation**: the position-restriction fix ruled out "cross-position
attention dilution" as the reason Phase 4 came back negative -- with that
confound removed, the result is still negative, just with a technically-
correct-signed but practically negligible correlation instead of a
wrong-signed one. This strengthens (not weakens) Phase 4's original
conclusion: the specific linear direction the probe reads from does not
appear to be causally load-bearing for the target's actual verification
computation, at least not via mean-ablation of this direction, at this
layer. Combined with Phase 3's clean AUROC results, the overall picture
remains "decodable but not shown to be the causal mechanism."

## Causal check — final summary

**What was tested**: whether the Phase 3 probe's layer-24 linear direction
is *causally load-bearing* for the target model's actual verification
computation (not just correlated with/predictive of it) -- via mean-ablation
of that direction with a random-direction control matched on mechanics.

**Phase 4 (diffuse, whole-sequence ablation)**: negative. Pearson r =
+0.028 (p=0.022) -- wrong sign versus the hypothesis's predicted negative
correlation. Effect size (mean |delta p(x)| = 0.0031) barely
distinguishable from the random control (0.0025). Code domain showed the
random control with a *larger* effect than the real probe direction.

**Phase 4b (position-restricted ablation, addressing Phase 4's main
caveat -- that ablating every sequence position at once could dilute a
real localized effect)**: sign flipped to the hypothesis-predicted
direction and reached significance on Spearman (rho = -0.033, p=0.007,
vs. random control's rho = -0.001, p=0.94), but Pearson stayed
non-significant (r = -0.016, p=0.204), effect size remained negligible
(mean |delta p(x)| = 0.0028 vs. control's 0.0023 -- same order of
magnitude as Phase 4), and the code-domain inversion persisted (random
control r=-0.056, p=0.009 vs. probe direction r=-0.019, p=0.373).

**Net conclusion**: ruling out cross-position dilution did not reveal a
hidden strong effect -- this constitutes a well-controlled negative
result, not an inconclusive one. The probe's layer-24 direction is
linearly decodable and predictive of accept/reject outcomes (Phase 3) but
is not shown to be causally load-bearing for the target's own
verification computation, under either ablation design tested (Phase 4 +
Phase 4b).

**This is the final result for the causal-check thread.** No further
ablation variants (different layers, interchange/activation patching, or
otherwise) are planned unless explicitly requested later.

## Open questions for user
- Confirm or override the EAGLE-vs-independent-drafter decision (Phase 0).
- Confirm Qwen2.5-7B/0.5B pair, or prefer a different target model family.
- Confirm no-KV-cache tradeoff (Phase 1) is acceptable, or prioritize adding
  caching now if throughput at 100k tokens looks likely to be a problem
  (Phase 2 ran in 35 min at this scale, so likely a non-issue unless
  scaling to millions of tokens later).
- Confirm chat-domain first-turn-only simplification (Phase 2) is
  acceptable, or want multi-turn MT-Bench prompts in a future data pull.
- Confirm pooled-domain probe training (Phase 3) is the right default, or
  want per-domain probes trained as a comparison.
- Confirm layer 24 + MLP as the target for Phase 4's causal check, or
  prefer a different layer/probe choice.

## Blockers
- None currently.

## Incident log (not a project blocker, but worth keeping a record of)
Mid-Phase-2/3, while troubleshooting SSH connectivity from a network that
blocks port 22, an attempt to add a secondary SSH port (9000) via a
systemd socket override on the H200 node failed partway through
(`systemctl restart ssh.socket` returned a failed job) and left the node
with **no SSH listener on any port** -- affecting all tenants of that
shared box, not just this project. Root cause: `ssh.socket` is
systemd-socket-activated, so `sshd_config`'s `Port` directive doesn't
control the actual listening ports; a socket-unit override is required,
and restarting an already-active socket unit has a stop/rebind race that
failed here. Recovered via the machine's out-of-band console (not SSH,
since SSH itself was down) once the node's admin (Tarang) applied the
revert. No data loss, no other tenants' running jobs affected (only new
connections were blocked). Full command sequence and revert steps are in
git history / this session's transcript if ever needed for a postmortem.
