# Tier 0 NOTES — surprises, bugs, judgment calls

## Step 0 finding (before any new code was written)
The existing Phase 3 probe (0.86 AUROC peak, layer 24) and the Phase 4/4b causal
ablation both read the hidden state at `pos = n - 1 + i` in
`speculative/sd_loop.py:110` — the position holding `x_{i-1}`, which under causal
masking has **never seen the drafted token `x_i`**. This is exactly "P_dec" in
this spec's terminology, not "P_tok". Implication: the existing probe was
structurally incapable of reading draft/target agreement; the Phase 4 causal
null is partially structurally guaranteed rather than purely empirical. See E2.

## Bugs found and fixed during pipeline construction
1. `datasets` library's legacy `"wikitext"` repo id no longer resolves (dead
   HF Hub script format) — fixed to `Salesforce/wikitext`.
2. MBPP dataset id: `google-research-datasets/mbpp`, config `"full"`, split
   `"test"` (500 examples) — not the bare `"mbpp"` id.
3. Logit-lens dtype crash: model is bf16, captured hidden states were being
   cast to float32 before `model.norm`/`lm_head`, which errored on a bf16
   weight matrix. Fixed by casting to `lm_head.weight.dtype` inside
   `logit_lens_batch` instead of forcing float32 upstream.
4. My own `pkill -f 15_tier0_generate.py` matched its own command line
   (contains the search string) and killed the SSH session running it —
   classic `pkill -f` self-match footgun, not a repo bug. Worked around with
   `pkill -f "[1]5_tier0_generate.py"`.
5. **Real engineering gap, not just a typo**: the first version of
   `scripts/15_tier0_generate.py` capped generation by *prompt count*, not
   *event volume*, and never cycled passes. Domains with few unique prompts
   (chat: 80 MT-Bench prompts) would have silently produced far less than
   the 150k-event target with no warning. Rewrote with budget-based
   stopping + pass-cycling, matching the original Phase 2 script's pattern.
6. Process-management mistake: launched the ~6-hour full-scale drafter-A
   generation as a foreground SSH command (via the harness's background-bash
   feature) rather than `nohup ... & disown` on the remote side. When the
   node had a transient full network outage mid-run, I could not tell
   whether the remote process had been killed (no `nohup` protection against
   SIGHUP) until the node came back and I could inspect the manifest —
   it turned out to have finished successfully, but this was luck, not
   good process hygiene. All subsequent long-running jobs (E1, drafter-B
   generation) were launched with proper `nohup ... & disown`.

## Judgment calls
1. **T=0 label/feature semantics**: all probability-based features (Group A
   logit-lens, Group B drafter q/entropy/margin) are computed at
   `feature_temp = decode_temperature if decode_temperature > 0 else 1.0`,
   since a literal T=0 softmax is degenerate (one-hot). Token selection and
   the accept/reject decision at T=0 still use pure argmax/argmax-equality,
   not this feature-only softmax. This keeps "which token was picked" and
   "how confident was the underlying distribution" as separate concerns, and
   makes Group A/B features comparable across the T=0 and T=0.7 conditions.
2. **No pass-cycling at T=0**: greedy decoding is a deterministic function of
   (prompt, argmax), so re-visiting the same prompt in a second pass produces
   byte-identical duplicate rows, not new information. Confirmed T=0 domains
   are reported at their true single-pass-achievable volume rather than
   padded with duplicates. This is why chat/T=0 has only 18,887 events (80
   unique MT-Bench prompts × achievable events/prompt), far short of 150k —
   documented as an honest data limitation, not a bug. All other 5
   domain×temperature combos hit ≥99.6% of the 150k target.
3. **Cheap-model (M_A/M_AB/M_ABC) training target**: plain sklearn
   `LogisticRegression`, trained directly on the realized binary `accepted`
   outcome rather than the soft `min(1,p/q)` label the existing pipeline
   (and M_h/M_full/M_offset here) uses — sklearn doesn't take soft targets,
   and a custom soft-label fit for these small cheap models added
   complexity without a comparability gain, since every model (cheap or
   not) is evaluated against the same realized-outcome test AUROC either
   way.
4. **Split assignment**: switched from a shuffle-and-slice split (like the
   existing `probes/splits.py`) to a **deterministic hash of
   `(seed, domain, idx)`** (`probes/tier0_splits.assign_split`). This was
   necessary because drafters A/B(/C) run independent budget-cycling random
   walks over each domain's prompt list and don't necessarily touch the same
   observed prompt subset — a shuffle-based split computed from "prompts A
   happened to touch" wouldn't give B a well-defined split assignment for
   prompts A never generated. The hash-based split makes split membership a
   pure function of the prompt itself, so any drafter's run agrees with any
   other's without needing identical coverage. Trade-off: exact 70/15/15
   proportions are only approximate now (matters most for chat's 80-prompt
   pool), not exact by construction.
5. **E3 drafter-B/C generation scope**: T=0.7 only (not both temperatures),
   ~40,000 events/domain (not matching A's 150k), `max_new_tokens=128` (not
   256) — pre-authorized by the coordinator as a disk/time-budget call.
   Consequence: E3 results are reported at T=0.7 only; no T=0 transfer
   numbers exist.
6. **Drafter C dropped**: `meta-llama/Llama-3.2-1B-Instruct` is a gated HF
   repo; the node has no HF token with accepted-license access, so loading
   it 401s immediately. This is a harder stop than the spec's own >15%
   cross-tokenizer-drop-rate escape hatch — we never got far enough to
   measure a drop rate. E3 is reported for drafter B only.
7. **E3 chat transfer_ratio > 1.0 is a data-starvation artifact, not "transfer beats retraining"**: chat's drafter-B train split only has 6,940 events (vs ~28,000 for code/reasoning, itself already reduced from A's 150k) because chat has only 80 unique MT-Bench prompts and B's generation doesn't cycle for chat's small pool either. The "retrained upper bound" model trained on that small set is undertrained and sits close to 0.5 AUROC, making the transfer_ratio ratio's denominator small and inflating the ratio (1.11–1.24) mechanically — read this as "both the frozen-A probe and the from-scratch-on-B probe are mediocre on chat, and mediocre-vs-mediocre ratios are noisy," not as genuine super-transfer. Code/reasoning transfer ratios (≈1.02–1.04, both positions) are the trustworthy part of this experiment's outcome and should be weighted more heavily.
8. **E1 verdict computed at the peak M_h(MLP)-AUROC layer**, pooled across
   domains for training with per-domain slicing for reporting — mirrors
   Phase 3's own pooled-training/per-domain-report convention, applied
   separately per temperature (T=0 and T=0.7 are fundamentally different
   regimes, not pooled together).
9. **LR-test numerical rough edge**: the MLP-variant LR stat came back NaN
   for T=0.0's peak layer (linear variant was fine: stat=-1062, p=1.0,
   i.e. M_offset(linear) log-likelihood was actually *worse* than M_ABC's,
   consistent with the negative ΔAUROC seen for the linear offset model).
   Root cause not tracked down given the AUROC-based decision rule (the
   actual decision criterion) was unaffected and unambiguous either way;
   likely a numerically-unstable probability near 0/1 from the MLP offset
   model on a handful of test rows. Flagging rather than silently
   re-running, per the "surface bugs, don't quietly work around them" rule
   — a full LR-test rerun would cost ~2.5h of compute for a supplementary
   diagnostic that doesn't change any conclusion.
10. **LR test (M_offset vs M_ABC) caveat**: degrees of freedom equal the
   trainable-parameter count of the offset model (3,585 for the linear
   variant on d=3584; far more for the MLP). At this many parameters versus
   even a few hundred thousand test rows, the classical chi-square LR test's
   asymptotics are strained and it will tend to report significance for
   almost any nonzero improvement — it's reported as the spec requires, but
   the **decision rule uses the bootstrap ΔAUROC CI**, not the LR test, for
   exactly this reason.
