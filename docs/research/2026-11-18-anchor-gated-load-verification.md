# Anchor-gated load verification, distilled from spliceware (2026-11-18)

Processed note. Raw source: [2026-11-18-spliceware-sacred-splicer-raw.md](2026-11-18-spliceware-sacred-splicer-raw.md) (Turns 13-22, proposals). Captured 2026-11-18 from `Moissanite-Labs/spliceware` at `7a1a27fec1c9ad886aaac0468846b4a6965e20de` and `sacred-splicer` tip `97b8feb135605f0570b7537fc5a4fd49f68d4816`. Mapped to **Q3** (can a cheap gate catch a malicious delta?), with direct input to **Q1** (does null-space projection hold on a real checkpoint?).

## Question

Can an O(K) load-time behavioral anchor gate plus static structural checks catch degenerate (HALT-collapse) or capability-breaking deltas cheaply, where null-space projection is either too costly at load time or blind to multi-step procedural regressions?

## Method

Comparative architectural analysis of the spliceware corpus's anchor machinery (Turn 21's projection trade-offs table, Turn 22's Triangular Sum analysis, and the two-tiered Lexicase gate in `proposals.md`), against [2026-11-14-alphaedit-and-scwsp.md](2026-11-14-alphaedit-and-scwsp.md) and [2026-11-14-compositor-prototype.md](2026-11-14-compositor-prototype.md). A toy numpy model ([prototypes/2026-11-18-lexicase-anchor-gate.py](prototypes/2026-11-18-lexicase-anchor-gate.py)) exercises the gate logic on a 16-input, 4-logit classifier with rank-12 preserved keys in a 16-dimensional input space.

## Evidence

`verified` (rerunnable on this repo, numpy 2.5.4, seed 20261118):

- The prototype's null-space check reproduces the compositor result shape: honest null-space delta leakage measures `5.72e-16` (small) and `5.89e-15` (large) on `||dW @ K0.T||_F`, matching the 3.7e-15 class of leakage reported for the toy compositor.
- A targeted rank-4 flip attack with `||dW||_F = 1.798e+00`, far under the budget (`9.274`), drops anchors to `1/4` and shows `||dW @ K0.T||_F = 5.492e+00`. It passes a norm-only gate and fails both the anchor gate and the null-space gate.
- A stepping-stone delta at `||dW||_F = 3.811e-02` flips exactly one anchor: `3/4`, refused by Tier 1 (4/4 required), admitted by Tier 2 (>= 3/4).

`source claim` (stated in the spliceware session, not reproduced on a real checkpoint):

- Pure fuel or norm minimization without anchor gating collapses immediately to trivial `0x00 HALT`, because the empty program burns zero fuel and wins every comparison.
- Rigid single-tier gates stall on multi-step procedural edits unless paired with a Tier 2 (>= 3/4) stepping-stone pool.
- The Triangular Sum ladder `(0, 0), (5, 15), (13, 91), (37, 191)` resists straight-line unrolling: no fixed `ADD` chain satisfies both `n = 5` and `n = 37`, and `(0, 0)` gives identity programs an immediate foothold to feed gradient into adaptive priors.

`proposed` (new to this note):

- Translate Two-Tiered Lexicase Anchor Gating to W-MCP low-rank deltas as K compact activation/logit canary probes bundled in `_meta` alongside null-space residual certificates. Tier 1 (canonical promotion) requires 4/4 anchors; Tier 2 admits >= 3/4 into a stepping-stone lane for re-evaluation, never into the canonical lane.
- Host-side load-time check sequence: verify checkpoint digest, verify structural bounds (rank and norm), then K canary forward passes, each bounded by a hard interaction budget.

## Result

On the toy, a norm-only certificate waves a targeted rank-4 attack through (`1.798` vs budget `9.274`) while the same attack fails both the anchor gate (`1/4`) and the null-space check (`5.49`). Honest null-space deltas pass all three gates at small and large norm. The Tier 1/Tier 2 split behaves as specified: `3/4` deltas are refused canonically and preserved as stepping stones.

One honest surprise from the toy. Earlier anchor sets drawn from the closest near-ties degraded under *any* nonzero delta, including honest null-space deltas, so an anchor gate on near-tie inputs would reject honest work. Anchor selection must span a difficulty ladder rather than cluster at the boundary, which is exactly why the spliceware proposal uses a spread `(0, 5, 13, 37)` instead of four hard cases.

This does not touch Q3's real question. Nothing here says anything about real transformer activations, multi-step reasoning regression, or a malicious delta engineered against the gate designer's anchor set. The toy demonstrates the mechanism, not the security property.

## Status

Open (Q3 and Q1 stay open). This note adds a mechanism and a toy, not an answer. The next real test is the AlphaEdit construction on a named checkpoint with canary probes bundled in the delta header.

Falsification criteria:

- Disproved for W-MCP if an adversarial low-rank delta passes K <= 8 canary forward passes and Frobenius norm bounds while still inducing more than 5% capability degradation on held-out reasoning benchmarks.
- Disproved if canary forward passes exceed host load-time latency budgets compared to matrix-only spectral checks.
- Disproved for the spliceware gate itself if `4/4` Triangular Sum anchors plus two-tier promotion fail to block `HALT` collapse on a real C11 substrate run with a live tournament (the spliceware C11 substrate is untested for this; it compiles and runs, but no experiment has been run).

## Next step

Q1: build the protected space from a real open-weights checkpoint and measure `||dW @ K0.T||_F` with and without canary anchors bundled. Q3: threat model an untrusted packet with a spectral norm cap, a protected-subspace overlap test, a checkpoint digest check, and K canary probes; record false positives and false negatives. The spliceware code in `substrates/c11/v3_evogenesis.c` and `v4_unigenesis.c` shows the in-packet anchor verification pattern the canary bundling would mirror.
