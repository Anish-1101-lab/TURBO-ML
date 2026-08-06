# Progress Log — Hidden-State Acceptance Probes for Speculative Decoding

## Status: Phase 2 complete, awaiting go-ahead for Phase 3

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

## Open questions for user
- Confirm or override the EAGLE-vs-independent-drafter decision (Phase 0).
- Confirm Qwen2.5-7B/0.5B pair, or prefer a different target model family.
- Confirm no-KV-cache tradeoff (Phase 1) is acceptable, or prioritize adding
  caching now if throughput at 100k tokens looks likely to be a problem
  (Phase 2 ran in 35 min at this scale, so likely a non-issue unless
  scaling to millions of tokens later).
- Confirm chat-domain first-turn-only simplification (Phase 2) is
  acceptable, or want multi-turn MT-Bench prompts in a future data pull.

## Blockers
- None currently. (Separately, mid-session: college wifi was blocking
  outbound port 22 to the remote nodes; user is setting up Tailscale as a
  more robust access path. Not a project blocker -- SSH access still works
  from networks that don't block port 22.)
