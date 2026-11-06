# E22 — mixing ladder at 3.2M and 10M CRPs (pre-specification DRAFT v2, 2026-11-06)

Status: v2 APPROVED by the author 2026-11-06, after the oracle design gate (PASS with required changes, applied;
experiments/logs/e22-design-gate.md). Nothing has been run. Once approved, this text becomes the docstring of
`experiments/encoding-induced-false-security-pufs-e22mixscale.py`, and the author commits it before the first run.

## Question
E18 found the E11 mask-table learner at chance (≤0.515) on repeated w=4 and w=8 mixing up to 1M CRPs, and only partly learning
w=2 at 1M (best-of-5 6/10). At w=2 the first successes appeared at the top of the grid, so the w≥4 failures are censored by
the data budget (oracle O5). E22 asks: **does the same frozen learner learn w=4 (and w=8) with more data, and does w=2 reach the
cost criterion?** Under the frozen fit rule the number of training steps grows with N, so every claim is about data and steps
growing together (28 epochs / 6,160 steps at 1M; 20 epochs / 14,080 steps at 3.2M; 20 epochs / 43,960 steps at 10M).

## Design (frozen learner, no new variant: consistent with the E17 stop rule, as in E18)
- Layers: E18's repeated-block layers mix2, mix4, mix8 (mix8 = E11 L_LIN), loaded through E18's own code, unchanged.
  No independent arm, no w=1 (already solved at 200k).
- Learner: E11 mask-table learner (`MaskModel('shared')`, E11 `fit()`, Adam LR 0.02, batch 4096, ≥6000 steps / ≥20 epochs,
  BCE, 10% validation), exec'd unchanged. Coordinate ascent is NOT run (≈1000 s per run at 1M, ≈44 GPU-h at 3.2M; out of scope).
- Instances: 30–39, the E18 block (paired with E18 on the N axis; fixed before E18, nothing selected by outcome; the test set
  (rng 12+100s, 10k) is identical at every N).
- Restarts: 0–4, seeds 1000*s + r (E18 convention). Primary: validation-selected best-of-5; best-of-3 reported next to it.
- Grid: N = 3,200,000 for rep2, rep4, rep8 (150 runs); N = 10,000,000 for rep4 only (50 runs). 200 runs in total.
  Data path: E11 `make_data` (challenge rng 11+100s train / 12+100s test, 10k test, validation = last tenth of N).
- Known-table control: not rerun; E18's K1 gate (10/10 at 10k on every rung, these instances) and its known N-to-0.95 medians
  (rep2 670, rep4 652, rep8 673) are the controls and denominators.
- Success = test accuracy ≥ 0.95 (primary; 0.90 secondary) of the validation-selected restart.

## Go-checks (before the main grid)
- G1 reproducibility: re-run E18 e11 rep2 instance 38 restart 3 at N = 1M through the E22 code path (E18 test accuracy 0.9962);
  require ≥ 0.95 and |Δ| ≤ 0.005. Assert that the first 1M rows of the 3.2M (and 10M) training draw and both test arrays equal
  E18's exactly. Failure → stop and debug.
- G2 timing pilot: one run at 10M on rep4, instance 60, seed 777000 (not a grid seed); log elapsed time, peak GPU memory and
  CPU RAM; accuracy left blank (not a result). Replaces the runtime estimate below before the grid starts.

## Predictions (author's priors in brackets; approved by the author 2026-11-06, R1 raised from 0.55 to 0.70 by the author)
- R1 rep2 at 3.2M: best-of-5 solves ≥ 8/10 (w=2 reaches the cost criterion).                                   [0.70]
- R2 rep4 at 3.2M: best-of-5 solves 0/10 and every selected run stays ≤ 0.55 test accuracy.                    [0.65]
- R3 rep8 at 3.2M: best-of-5 solves 0/10 (replicates E11/E18 at 3.2× the data).                                [0.85]
- R4 monotone: solved count at 3.2M non-increasing in w (rep2 ≥ rep4 ≥ rep8).                                   [0.85]
- R5 rep4 at 10M: best-of-5 solves 0/10 and every selected run stays ≤ 0.55 test accuracy.                     [0.50]
The 0.55 band is new (E18 used 0.95/0.90 only): chance has a standard error of about 0.005 on 10k test rows, and E18's rep4
maximum was 0.515.

## Claims by outcome (fixed now; the paper uses the matching wording)
rep4 (judged at the largest N with a success, else at 10M):
- ≥ 8/10 at some N   → four-byte mixing is learnable by this learner at that N (data and steps growing together); informed-cost
                        ratio N / 652 (grid-bounded upper bound); thesis scope extends to w ≤ 4; abstract, discussion and
                        limitations are rewritten; w=8 status as measured.
- 1–7/10 at some N   → four-byte mixing is partly learned at that N (counts reported), no cost; "open" becomes "partly learned".
- 0/10, all ≤ 0.55   → the learner bound moves from 1M to 10M CRPs; w ≥ 4 stays open, not secure (no change in claim type).
- 0/10, some selected run in (0.55, 0.95) → reported as measured: above chance but unsolved; no claim beyond the counts.
rep2 at 3.2M:
- ≥ 8/10             → w=2 cost = 3.2M / 670 (≈ 4800×, single grid point, upper bound).
- < 8/10             → two-byte mixing partly learned at 3.2M (counts reported), no cost; a count below E18's 6/10 at 1M is
                        reported as measured.
rep8 at 3.2M:
- 0/10               → the learner bound on the full linear layer moves to 3.2M CRPs.
- any success        → reported as measured; it contradicts the E11/E18 record on the same layer at lower N and is checked against G1.
Any restart that reaches 0.95 but is not the validation-selected one → reported as exploratory, not counted.
Single-seed instability or a non-monotone count is reported as measured; no instance is dropped; nothing is retuned.

## Kill criteria
- K1: G1 fails → stop (code path not comparable with E18).
- K2: G2 shows the 10M run cannot fit in GPU memory or CPU RAM without changing the learner → drop the 10M point, report it, run
  the 3.2M grid; no learner change.
- No other stopping rule: all runs are reported.

## Environment and device
GPU: NVIDIA RTX 4070 Laptop (8 GB), `puf-clean` conda env, torch 2.8.0+cu129, pypuf 2.2.0 (pinned, unchanged). TF32 off,
CPU-side init/batch order as in E18. The feature tensor is built per batch and accuracy is computed in 20k-row chunks (E11 code),
so GPU memory is not expected to limit; the 10M training tensor is about 640 MB of int64.

## Runtime estimate (inferred from E18 timings and the fit rule, not measured)
E18 e11 at 1M: median 17 s per run (6,160 steps). 3.2M: 14,080 steps (2.29×) ≈ 40 s per run, 150 runs ≈ 1.7 h. 10M: 43,960 steps
(7.1×) ≈ 125 s per run, 50 runs ≈ 1.7 h. Total ≈ 3.5 h plus label generation (not included in E18's elapsed_s). G2 replaces this.
Long-run protocol: detached, progress file + exit-code file in experiments/logs/.

## Analysis plan
Summary per rung and N: solved counts (0.95 primary, 0.90 secondary) for best-of-3 and best-of-5; max and median selected test
accuracy; R1–R5; per-instance table paired with E18's 60k/200k/1M results. Anything else is labelled exploratory.

## Smallest effect that would matter
Any rep4 success (≥1/10 at ≥0.95) changes the paper's w=4 wording from "open (at chance to 1M)" to "partly learned".
