# Homologous composition and adaptive priors, distilled from spliceware (2026-11-18)

Processed note. Raw source: [2026-11-18-spliceware-sacred-splicer-raw.md](2026-11-18-spliceware-sacred-splicer-raw.md) (Turn 20, `domain.md` / `code-13`, Turns 16-17). Captured 2026-11-18 from `Moissanite-Labs/spliceware` at `7a1a27fec1c9ad886aaac0468846b4a6965e20de`. Mapped to **Q4** (how do concurrent adapters avoid overwriting each other?).

## Question

When several adapters target the same base checkpoint, what stops them from overwriting or diluting each other's capabilities, without a destructive, last-write-wins merge?

## Method

Architectural analysis of the spliceware corpus's composition mechanisms: `sacred_splice()` homologous crossover (Turn 17, `code-07`/`code-08`), symmetric Mitosis A/B tournaments (Turn 16, `code-05`/`code-06`), in-cell adaptive priors (Turn 20, `code-13`), and the seam verification in `substrates/c11/v2_adaptive_loop.c`. Reference points: [2026-11-14-taste-marketplace.md](2026-11-14-taste-marketplace.md), [2026-11-14-compositor-prototype.md](2026-11-14-compositor-prototype.md).

## Evidence

`source claim`:

- Crossover is permitted only at PC positions where cumulative stack depth returns to zero (`Δsp == 0`), verified by the load-time scan (`verify_and_map` in v2, `Opcode Valence Table` in `code-08`). The effect is a composition rule: graft a donor segment only at points where neither side's stack contract is violated.
- Mitosis (`0xFFFF`) forks execution into `ancestor_a` and `ancestor_b` branches and runs a symmetric A/B fuel tournament. Both branches pay the same cost model, so the winner reflects the splice, not a handicap.
- Adaptive priors (`HDR_WEIGHTS`) travel inside the delta itself. Codons that win tournaments are boosted multiplicatively (`adapt_prior()`), so a composed adapter's sampling bias moves with it, not in an external controller.

`verified`:

- The v2 substrate compiles clean and runs: `verify_and_map` rejects unbalanced `LOOP..END` at load time, which is the load-time analogue of refusing a delta that violates a structural bound.

`proposed`:

- Translate "homologous seams" to adapter composition: admit an adapter only at boundaries where its interference with already-mounted capabilities is measurably zero (residual inner product below a threshold, or a null-space-style orthogonality certificate). Last-write-wins is the default failure mode; seam admission is the correction.
- Symmetric A/B tournaments for concurrent adapters: score composed-vs-incumbent pairs on anchor retention plus a metabolic budget, never on the tournament alone deciding contested human preferences.
- Portable meta-priors as part of the delta metadata: a small, bounded prior vector the host reads to bias future composition, analogous to `HDR_WEIGHTS`.

## Result

The corpus contributes structure, not a measured answer. Every mechanism in it is a candidate composition rule for Q4, and none has been run against real checkpoints or even against the MDE toy. The strongest transferable idea is the gate: composition happens only at verified-zero-interference seams, with A/B shadow evaluation before commit. The weakest is adaptive priors; they assume a search process exists to consume them, which weight merging does not have.

Non-negotiable bound: none of this resolves contested human preferences. Veto 2 applies in full. Tournaments here rank algorithmic candidates within Plane 2; they never arbitrate "which refusal behavior should win" or any ethos question.

## Status

Open (Q4 stays open). Needs the allocation rule, a composed-N experiment on a named checkpoint, and a regression curve as N grows.

Falsification criteria:

- Disproved if subspace/seam-constrained adapter composition underperforms simple weighted LoRA averaging (ties/dare) under equal rank and evaluation compute on real open-weight checkpoints.
- Disproved if seam verification rejects most honest adapter pairs (a gate that blocks honest work is a tax, not a protection).
- Disproved as a concurrency answer if deadlock or starvation appears between two adapters both seeking the same seam.

## Next step

Build the allocation rule for one named open checkpoint: mount adapters sequentially with seam certificates, record per-adapter regression as N grows, compare against weighted averaging at equal rank.
