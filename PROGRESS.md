# Progress Log — Hidden-State Acceptance Probes for Speculative Decoding

## Status: Phase 0 complete, awaiting go-ahead for Phase 1

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

## Next
- Phase 1: instrumented SD loop (draft -> parallel verify -> accept/reject/
  resample), with per-token hidden-state logging at a layer stride (TBD,
  will justify against memory budget), validated by hand against a manually
  computed reference on a handful of prompts.

## Open questions for user
- Confirm or override the EAGLE-vs-independent-drafter decision above.
- Confirm Qwen2.5-7B/0.5B pair, or prefer a different target model family.

## Blockers
- None currently.
