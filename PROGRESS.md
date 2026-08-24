# Progress Log — Hidden-State Acceptance Probes for Speculative Decoding

## Status: Phase 5 COMPLETE (steps 1-3 done at threshold=0.995). Verdict: mixed/negative -- 1.02x median overall speedup (code +8% median, reasoning/chat slower), still 5% overall completion-divergence rate. Not a general systems win. Stopped per instruction; no further Phase 5 work without explicit direction.

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

## Phase 4c: interchange patching

**Motivation.** Phase 4/4b's stated limitation was that mean-ablation is a
blunt intervention -- it destroys the probe direction's information toward
a population-average value rather than substituting a specific, real
alternative. Interchange patching is a strictly stronger test: splice a
REAL donor example's own activation into a recipient's forward pass, and
check whether the recipient's p(x) shifts toward the donor's actual
outcome. Phase 4/4b's code, data, and PROGRESS.md sections above are
unmodified by this phase.

**Design.** Key implementation observation: interchange-patching a single
direction is mechanically identical to Phase 4/4b's mean-ablation of that
direction -- both replace the component of a hidden state along a unit
direction with a target scalar, leaving orthogonal components untouched.
The only difference is what the target scalar is (a real donor's
projection, not the Phase 3 population mean). This let Phase 4c reuse
`probes/ablation.py`'s `AblationHook` completely unchanged (`mean_proj` is
just set per-patch instead of fixed at construction) -- new code
(`probes/interchange.py`, `scripts/12_interchange_patching.py`,
`scripts/13_analyze_interchange.py`) is confined to data collection and
analysis. Position-restricted patching only (Phase 4b's stronger design).

A fresh generation pass was required (not Phase 4/4b's saved
`causal_records.json`) because interchange patching needs each candidate's
full token context to re-run a patched forward pass, and Phase 4/4b's
saved output only persisted scalar metrics, not token sequences. Same 60
prompts (20/domain), same sampling/seed convention as Phase 4/4b, driven
by the real, unmodified accept/reject rule.

**Pairing**: per prompt, after its generation completed, the single
highest-label record (>= 0.9, "near-certain accept") and single
lowest-label record (<= 0.3, "near-certain reject or borderline") from
that SAME prompt's own trajectory were paired -- same domain and same
underlying prompt by construction. **59 of 60 prompts yielded a
qualifying pair** (one chat prompt's minimum label was 0.393, above the
0.3 cutoff) -- n=59 pairs, plainly smaller than Phase 4/4b's ~6700
per-token-record sample, an inherent consequence of a matched-pair design
(one pair per prompt, not one record per drafted token).

**Result: still negative, and more informatively so once a data-quality
issue in the pairing was accounted for.** Per pair, per direction (probe,
random-matched-to-Phase-4/4b), two swap directions were measured:
patching the HIGH donor into the LOW recipient's own position (predicts
p(x) should rise), and patching the LOW donor into the HIGH recipient's
own position (predicts p(x) should fall).

*high-into-low* turned out to be **largely uninformative, not just
null**: because pairs were selected for extremity, LOW recipients'
clean p(x) is itself already deeply floor-saturated (median 4.5e-7 across
the 59 pairs; 46/59 below 1e-4) -- there is essentially no room for
probability to rise further in absolute terms regardless of whether the
intervention matters causally. Both probe (mean |shift| = 0.00011) and
random (mean |shift| = 0.00004) shifts are correspondingly tiny, and a
paired test shows no distinction (Wilcoxon p=0.826). This swap direction
should be read as a limitation of extremity-based pairing, not as
evidence against causality.

*low-into-high* has real dynamic range for a genuine (if not universal)
subset of pairs -- HIGH examples' clean p(x) has median 0.995, but 13/59
pairs sit below 0.9 and 29/59 below 0.99, giving real room to observe a
decrease for roughly half the sample even though the other half is
itself close to ceiling-saturated (a spot-check of pair 0, code domain,
found p_clean_high=0.999986 and p_clean_low=4.6e-11 -- saturated at both
ends, and correspondingly showed almost no shift in either direction,
consistent with the aggregate pattern rather than contradicting it) -- and
shows genuinely larger shifts in aggregate -- mean
|shift| 0.00506 (probe) vs. 0.00308 (random), roughly an order of
magnitude bigger than the floor-locked direction. But **these shifts are
not reliably in the causally-predicted (downward) direction for either
condition**: only 44.1% of probe-direction patches and 33.9% of
random-direction patches move p(x) down at all -- both below the 50%
a coin flip would give, and a paired test finds no significant
probe-vs-random difference (Wilcoxon p=0.391, paired t-test p=0.559).
Probe's average signed shift (+0.00013) is closer to zero / less
wrong-signed than random's (-0.00051), a small directional edge for
probe, but this is not close to statistical significance at n=59 and
should not be read as support for the causal hypothesis.

**Per-domain (low-into-high, the informative swap direction)**: code
n=20, probe frac-correct 40.0% vs. random 25.0% (no inversion -- probe
less wrong than random here, unlike Phase 4/4b); reasoning n=20, probe
frac-correct 40.0% vs. random 45.0% (inversion -- probe *more* wrong-signed
than random on average here, the opposite domain from Phase 4/4b's
code-domain inversion); chat n=19, probe 52.6% vs. random 31.6% (no
inversion). None of the three domain-level paired tests approach
significance (p=0.23-0.63) at n~20/domain.

**Comparison against Phase 4/4b: confirms, does not overturn, and adds
detail.** A strictly stronger intervention (real donor substitution
instead of mean-ablation) still does not produce a reliable,
majority-of-pairs shift in the causally-predicted direction on the one
swap direction where the outcome variable had room to move -- if
anything, LESS than half of patches (for both probe and random
directions) go the predicted way, a more clearly negative headline
number than Phase 4/4b's near-chance correlations. The domain-level
inversion pattern is not stable across methods (code in Phase 4/4b,
reasoning in Phase 4c) -- read as evidence the small per-domain samples
(n~20) are noisy at this effect size, not as a reproducible domain
effect. **Flagged limitation for any future follow-up**: pairing at less
extreme thresholds (e.g. 0.7/0.3 instead of 0.9/0.3) would avoid the
floor-saturation problem that made the high-into-low swap direction
uninformative, and should be the first change to make before drawing
further conclusions from this specific design -- not attempted here.

## Causal check — final summary

**What was tested**: whether the Phase 3 probe's layer-24 linear direction
is *causally load-bearing* for the target model's actual verification
computation (not just correlated with/predictive of it) -- across three
independent designs: mean-ablation (whole-sequence and
position-restricted; Phase 4/4b) and interchange patching with real donor
examples (position-restricted; Phase 4c) -- each with a matched
random-direction control.

**Phase 4 (diffuse, whole-sequence mean-ablation)**: negative. Pearson r =
+0.028 (p=0.022) -- wrong sign versus the hypothesis's predicted negative
correlation. Effect size (mean |delta p(x)| = 0.0031) barely
distinguishable from the random control (0.0025). Code domain showed the
random control with a *larger* effect than the real probe direction.

**Phase 4b (position-restricted mean-ablation)**: sign flipped to the
hypothesis-predicted direction and reached significance on Spearman (rho
= -0.033, p=0.007, vs. random control's rho = -0.001, p=0.94), but
Pearson stayed non-significant (r = -0.016, p=0.204), effect size
remained negligible (mean |delta p(x)| = 0.0028 vs. control's 0.0023),
and the code-domain inversion persisted (random control r=-0.056,
p=0.009 vs. probe direction r=-0.019, p=0.373).

**Phase 4c (position-restricted interchange patching, real donor
examples instead of a population mean -- the strongest test applied)**:
negative, and on the swap direction with real dynamic range to observe an
effect in, *more clearly* negative than either mean-ablation design --
fewer than half of patches (44.1% probe, 33.9% random) move p(x) in the
predicted direction at all, with no significant probe-vs-random
difference (Wilcoxon p=0.391). A companion swap direction was confounded
by floor-saturated outcome probabilities and was uninformative rather
than negative.

**Net conclusion, now across three designs of increasing intervention
strength**: a progressively stronger causal test -- diffuse ablation,
then position-restricted ablation, then position-restricted interchange
patching with real donor substitution -- did not reveal a hidden strong
effect at any step. This constitutes a well-controlled negative result,
not an inconclusive one. The probe's layer-24 direction is linearly
decodable and predictive of accept/reject outcomes (Phase 3) but is **not
shown to be causally load-bearing** for the target's own verification
computation, under three independent ablation/patching designs.

**This is the final result for the causal-check thread**, updated to
incorporate Phase 4c rather than left stale. No further ablation or
patching variants (different layers, less-extreme pairing thresholds, or
otherwise) are planned unless explicitly requested later.

## Phase 5: verification-skipping gate (in progress)

### Framing (stated explicitly per user instruction)

This phase proceeds on **"predictive is enough for a gate" grounds only,
not because the mechanism is understood.** To restate the current
evidence honestly:
- Phase 3: the layer-24 MLP probe is a real but modest predictor --
  AUROC ~0.85-0.86, peaking a few layers before the final one (28), not
  dramatically early.
- Phase 4 + 4b: a well-controlled **negative** causal result. The probe's
  direction is not shown to be causally load-bearing for the target's own
  verification computation, under two different ablation designs.
- Nothing below should be read as implying a causal explanation for why
  the gate works when it works. It is an empirical shortcut that trades a
  small, measured risk of divergence (step 2) for a small, measured
  amount of compute (step 3) -- both numbers taken at face value, not
  assumed.

**Expected ceiling is modest, and not automatically achieved.** Skipping
layers 25-28 covers 4 of 28 decoder layers (~14.3% of depth). But this is
a ceiling on savings for an individual gated POSITION, not a guaranteed
per-round number: the standard SD algorithm already computes one shared
forward pass covering an entire round (context + all k drafted
positions) at once, and every round still needs at least one real
(non-gated) pass through all 28 layers regardless of gating -- either for
the "bonus" token sampled from the target's own distribution when every
drafted token in a round is committed, or for whichever position
actually triggers a real rejection. So a round only saves compute to the
extent gated positions let a LATER real-verification call be reached with
a SHORTER effective sequence length, or let it be skipped ahead of an
early real rejection -- not simply "each gated token times 4/28 always
subtracted." No incremental KV-cache is used anywhere in this project
(Phase 1 decision), so each real-verification call recomputes attention
from scratch over whatever length it needs; a round with several
non-gated positions in a row pays for overlapping recomputation rather
than an incremental extension. Net effect on real wall-clock is
genuinely unclear from the arithmetic alone -- that is exactly what step
3 is for. Report the measured number, not the theoretical one.

### Gate design

**Decision rule** (`probes/gate.py`): at each drafted position, compute
the layer-24 hidden state, standardize with the frozen Phase 3
train-split `mu`/`sigma` (never refit), run it through the trained MLP
probe (`analysis/phase3/probe_mlp_layer24.pt`), and take
`sigmoid(logit)` as the predicted acceptance probability. If that
probability is `>= threshold`, the gate fires: commit the drafted token
immediately, skip the layer-25-28 computation for that position entirely.
If it does not fire, fall back to the exact accept/reject rule exactly as
in the ungated baseline (`speculative/sd_loop.py`), computed via a real
(non-skipped) forward through the remaining layers.

**The gate can only pre-empt an accept, never decide a reject.** There is
no code path where the gate's output causes a rejection -- a rejection is
only ever produced by the real accept/reject draw, which by construction
only runs when the gate does not fire. Implemented in
`speculative/gated_sd_loop.py` (`gated_verify_and_step_shadow` for
validation, `gated_verify_and_step_fast` for real layer-skipping) and
`speculative/layer_skip.py` (the actual layer-truncation mechanics: swaps
`target.model.layers` for a shorter `nn.ModuleList` and reuses the real
HF forward code, rather than hand-reimplementing decoder internals --
see that file's docstring for the norm-placement and rotary-position
subtleties this requires getting right, and the explicit note that the
prefix+suffix composition is UNVALIDATED numerically as of this write-up).

**MLP probe, not linear** (`configs/gate_config.yaml`): Phase 3 found
linear probes poorly calibrated (test ECE ~0.25-0.28 across layers)
despite decent AUROC, while MLP probes are well-calibrated (ECE
~0.01-0.03). A threshold-based gate is only honest if "0.98 predicted
probability" really does correspond to ~98% empirical acceptance --
that requires calibration, which only the MLP probe has.

### Threshold choice: 0.98

Picked by inspecting Phase 3's saved calibration data
(`analysis/phase3/reliability.json`, `mlp_layer24_overall`), not by any
new run. The top calibration bin (predicted probability 0.933-1.0, n=7006
of 15559 test-set records, ~45% of the test set) has mean predicted
confidence 0.988 against an empirical acceptance rate of 0.983 -- i.e.
close to perfectly calibrated in aggregate at the top of the range.
0.98 was chosen as a cutoff inside the upper portion of that bin, on the
reasoning that restricting to the top of an already-well-calibrated bin
should, if the calibration trend holds, give an empirical acceptance rate
at or above that bin's 98.3% aggregate figure (since the bin's lower end,
0.933-0.98, likely pulls the aggregate down more than the upper end pulls
it up).

**Explicit limitation**: Phase 3's saved reliability data only has
15-bin resolution -- there is no directly-measured empirical acceptance
rate at exactly the 0.98 cutoff, only the coarser 0.933-1.0 aggregate.
This is an extrapolation from a calibration trend, not a directly
verified number. **Step 2's losslessness validation is the actual
empirical test of whether 0.98 is safe enough** -- this threshold choice
should be read as a starting point for that test, not as a validated
final answer.

### Layer-skip mechanism check (passed)

Before trusting `layer_skip.py` for anything, ran
`scripts/08_verify_layer_skip.py` on 3 real prompts on the H200 node:
`forward_prefix` matches a real `output_hidden_states=True` call's
`hidden_states[24]` exactly, and `forward_prefix` + `forward_suffix`
composed together (both full-length and length-restricted to an earlier
position) reproduces the real full-depth model's logits exactly (max abs
diff 0.00000, identical argmax) on every prompt tested. The mechanism is
trustworthy.

### Step 2: losslessness validation (result: NOT lossless -- real, non-trivial divergence)

`scripts/09_gate_losslessness.py`. 60 held-out prompts (20 each,
code/reasoning/chat, fresh sample, not reused from Phase 4/4b), 128 new
tokens/prompt, threshold 0.98. For each prompt, ran the ungated baseline
and the gated-shadow generation with the identical seed and compared.

**Two measurements, both reported plainly:**

| | overall | code | reasoning | chat |
|---|---|---|---|---|
| gate fire rate (frac. of positions gated) | 32.5% | 44.9% | 39.4% | 14.7% |
| local mismatch rate (gate said accept, real rule would reject \| gated) | 0.41% | 0.50% | 0.35% | 0.29% |
| **exact output match rate (60 prompts)** | **85%** (51/60) | 75% (15/20) | 85% (17/20) | 95% (19/20) |

The **local** mismatch rate is low and consistent with (if anything
slightly better than) the threshold's Phase 3 calibration-bin
extrapolation (that predicted ~1.7% from the raw 0.933-1.0 bin; 0.98 as an
even-higher cutoff giving 0.41% is directionally exactly what the
threshold reasoning in step 1 predicted).

**But the local rate is not the whole story, and should not be reported
as "the" result.** Generation is autoregressive: once a single gate
mismatch happens, every token after it is conditioned on a token the
ungated baseline would never have produced, so the two trajectories
diverge for the rest of the generation. Checking the actual data: **every
one of the 9 diverged prompts had exactly one gate mismatch, and every
prompt with zero gate mismatches matched exactly** -- a clean, fully
explained relationship, not noise. The consequence is that a 0.3-0.5%
per-position error rate, compounded over ~110-125 drafted positions per
128-token generation, produces a **15% chance that a given completion
comes out completely different from what the exact algorithm would have
produced** (25% for code specifically, the domain with the highest gate
fire rate).

**This is a genuine divergence, not noise or a rounding artifact, and
this gate as configured (threshold=0.98) is NOT lossless in any strict
sense.** Whether a 15% (up to 25% for code) rate of full-completion
divergence is an acceptable cost for whatever compute saving step 3
measures is a product decision, not a technical one -- stated here rather
than decided here, per instruction. Full per-prompt data in
`analysis/phase5/losslessness.json`.

### Step 2 rerun at threshold=0.995 (same 60 prompts, same seeds, same 128-token cap)

Single-parameter rerun (`scripts/09_gate_losslessness.py --threshold
0.995`, no changes to `gate.py`, `layer_skip.py`, or
`gated_sd_loop.py`), saved separately to
`analysis/phase5/losslessness_t0995.json` so both results stay on record.

| | threshold=0.98 | threshold=0.995 |
|---|---|---|
| gate fire rate -- overall | 32.5% | 24.5% |
| gate fire rate -- code | 44.9% | 36.6% |
| gate fire rate -- reasoning | 39.4% | 27.7% |
| gate fire rate -- chat | 14.7% | 10.2% |
| local mismatch rate -- overall | 0.41% | 0.18% |
| local mismatch rate -- code | 0.50% | 0.12% |
| local mismatch rate -- reasoning | 0.35% | 0.33% |
| local mismatch rate -- chat | 0.29% | 0.00% |
| exact-output match rate -- overall | 85% | 95% |
| exact-output match rate -- code | 75% | 95% |
| exact-output match rate -- reasoning | 85% | 90% |
| exact-output match rate -- chat | 95% | 100% |

**Reading the actual numbers rather than rounding toward either
preconception**: gate fire rate dropped by about a quarter (relative,
32.5% -> 24.5%), while the local mismatch rate roughly halved (0.41% ->
0.18%, -56% relative) and the full-completion divergence rate dropped by
about two-thirds (15% -> 5% overall, -67% relative). These are NOT
proportional -- divergence fell faster than the gate-fire rate did. That
is closer to case (a) than case (b): there is headroom in this
threshold's range where raising it buys a disproportionate reliability
improvement without the fire rate collapsing to near-zero, particularly
in the code domain (fire rate stayed at 36.6%, still over a third of
positions, while its divergence rate fell from 25% to 5%).

**This is not a clean pass, and should not be reported as one.**
Full-completion divergence at threshold=0.995 is still 5% overall (10%
for reasoning specifically, 1 case of 20) -- a real, nonzero rate of
producing different output than the exact algorithm, not zero. "Better"
is not the same as "acceptably lossless" and the choice of what counts
as acceptable is not being made here. What this rerun does establish is
that the gate design is not doomed by construction (case b is not what
happened) -- there is a threshold response worth taking to step 3 if the
user decides a low-single-digit-percent divergence rate, in exchange for
whatever compute saving materializes, is a trade worth measuring.

### Step 3: wall-clock benchmarking, threshold=0.995 only (final Phase 5 step)

Threshold=0.98 was not benchmarked -- its 15% divergence rate already
ruled it out. `scripts/10_gate_benchmark.py`, using the REAL
layer-skipping path (`gated_verify_and_step_fast` /
`run_gated_speculative_decoding_fast`, genuinely skips layers 25-28 for
gated positions), not step 2's shadow path. Same 60 held-out prompts,
same seeds, 128-token cap. Gate fire rate this run: 24.6% overall --
matches step 2's 24.5% at this threshold within noise, no bug indicated.

**Timing hygiene, stated plainly rather than left implicit**: NVIDIA H200
NVL, GPU index 4, idle at run start (other GPUs on this shared box were
in use by other tenants, index 4 was not). One short warmup generation
per model before timing, excluded from all numbers below. **One timed
run per prompt per condition -- no repeated trials of the same prompt**;
aggregate stats come from spread across 60 distinct prompts, not
resampling. Fixed order (ungated timed first, then gated) per prompt,
not alternated/randomized -- a real limitation on ruling out systematic
drift over the run. `torch.cuda.synchronize()` brackets each condition's
whole generation, not sub-steps within it.

**Result -- paired with the divergence rate every time, never stated alone:**

| domain | mean speedup | median speedup | gate fire rate | step 2 divergence rate (this threshold) |
|---|---|---|---|---|
| **overall** | **1.09x** | **1.02x** | 24.6% | **5%** |
| code | 1.34x | 1.08x | 37.0% | 5% |
| reasoning | **0.94x (slower)** | **0.92x (slower)** | 27.8% | **10%** |
| chat | 0.96x (slower) | 0.97x (slower) | 10.1% | 0% |

**Probe overhead** (isolated microbenchmark, not interleaved with the
generation timing above): 82.3 microseconds/call, 6740 total probe calls
across the whole run, summing to an estimated 0.55 seconds of probe
compute against 171.1 seconds of total gated wall-clock across all 60
prompts -- about 0.3% of gated runtime. **The probe itself is not what's
costing time.** The slowdowns in reasoning and chat come from the
mechanism flagged as a known limitation back in step 1's design write-up:
no KV-cache anywhere in this project, so the "fast" path's suffix calls
recompute attention from scratch each time real verification is needed,
and that redundant recomputation can cost more than the 4 skipped layers
save -- exactly the failure mode predicted, now measured rather than
theorized.

**Plain verdict, not rounded toward success:** this gate at threshold=0.995
produces a **net-negative or negligible result in two of three domains**
(reasoning: measurably slower AND still 10% divergent -- the worst
combination in the table; chat: measurably slower, though at least fully
lossless in this sample at 0%). **Code is the only domain with a real
speedup** (1.34x mean / 1.08x median), and even there the gap between
mean and median (driven by a small number of prompts, n=20, single run
each -- see timing-hygiene caveats above) means the 1.08x median is the
more trustworthy number, not the 1.34x headline. The overall/pooled
1.09x mean / 1.02x median is arithmetically dominated by code; it is not
evidence of a general win, and the median in particular (2%) is small
enough to be within the noise floor of a single-run-per-prompt
methodology.

**This does not support treating the gate as a systems contribution as
currently designed.** It is, at best, a domain-specific proof-of-concept
(code, ~8% median speedup, medium confidence given methodology) riding
alongside a domain where it actively regresses both speed and
correctness (reasoning) and a domain where it's lossless but not faster
(chat) -- all of this on top of a 5% overall completion-divergence rate
that was never resolved to zero. The honest reading is that the
per-position "skip 4 of 28 layers" savings this gate targets are smaller
than the cost of recomputing attention without a cache once any real
verification is needed in a round, in two of three domains tested. A
KV-cached suffix path might change this picture, but that is a new
experiment, explicitly out of scope for this closeout.

### Supplementary: why reasoning was slower and code was faster

Run to support the research report's discussion section
(`report/report.md`), not a new Phase 5 step -- `scripts/11_gate_round_analysis.py`,
same 60 prompts/seeds/threshold=0.995, no new mechanism code. Reconstructs
per-round composition (gated vs. real-verification positions) and where in
each completion gated positions fire.

**Bug caught before reporting, not before running**: the first version
inferred round boundaries after the fact from gaps in position numbers.
That silently merges consecutive rounds whenever a round ends in an
ordinary rejection (no position gap in that case -- only a round that
fully completes with a bonus token leaves one, since the bonus position is
never logged as a record). Rejections are the normal, frequent SD outcome,
not rare, so this wasn't a corner case -- e.g. chat's first (buggy) run
showed 8.75 "real verifications per round" against a hard cap of k=4 per
round, which is what caught it. Fixed by tracking round boundaries
directly during generation (manual per-round loop, not post-hoc
inference) and rerun before any numbers were used.

**Result** (`analysis/phase5/round_composition.json`): completion length
is essentially identical across domains (129.2-129.9 mean tokens) --
rules out "reasoning completions are just longer" as an explanation.
Two other factors both point toward code and away from reasoning,
compounding rather than either dominating alone: (1) code's gate fires
disproportionately late in the completion (52.8% of its firing events in
the last third, vs. 40.8% reasoning / 36.1% chat) -- skipping
disproportionately the most expensive potential real-verification calls;
(2) reasoning needs a real (uncached) verification event in 95.6% of its
rounds, more than code's 84.5%, and slightly more such events per round
when it does (2.46 vs. 2.05 mean). Chat fires so rarely (10.1%) that its
behavior stays close to the ungated baseline throughout, consistent with
its comparatively small measured slowdown. Full numbers and interpretation
in `report/report.md` Section 7.

### Status: Phase 5 complete. Verdict: mixed/negative -- not a general
speedup, domain-specific at best, still imperfectly lossless.

Per instruction, stopping here. No further threshold sweeps, no
KV-cache follow-up, no other Phase 5 variants without explicit direction.

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
