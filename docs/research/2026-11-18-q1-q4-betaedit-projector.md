# Q1 & Q4 Prototype: History-aware `BetaEdit` null-space projector and SGMV slab arena

Author: Gemini (machine-authored research note and reference prototype), 2026-11-18.

## Question

- **Q1 (P0) — Does null-space projection hold on a real checkpoint?**
- **Q3 (P0) — Can a cheap gate catch a malicious delta?**
- **Q4 (P1) — How do concurrent adapters avoid overwriting each other?**

Stage a clean, self-contained PyTorch reference module (`docs/research/prototypes/2026-11-17-betaedit-projector.py`) addressing the mathematical failure mode of static `AlphaEdit` null-space projection under sequential updates before running real-checkpoint evaluations.

## Method

Implemented `BetaEditNullspaceProjector` in `docs/research/prototypes/2026-11-17-betaedit-projector.py` (strictly isolated from `mep/` under the non-MEP reference banner) combining:
1. **History-aware covariance accumulation (`BetaEdit`, arXiv:2605.09285):** Tracks `C^(t) = lambda_1 K_0 K_0^T + sum_{i<t} K_i K_i^T` with periodic eigenspace projector refresh every `tau` commits.
2. **Exact block-triangular SPD subspace reduction:** Because `U^T (lambda_2 I + M P_t) U` is block lower-triangular in `[U_small, U_perp]`, left-multiplying its inverse by `P_t = U_small U_small^T` annihilates the off-diagonal block identically, reducing BetaEdit Eq. (9) from a non-symmetric `d_in x d_in` inverse to a symmetric positive-definite `m x m` Cholesky solve.
3. **Matrix-free Eckart-Young rank-`r` factorization & SGMV slab arena (`Punica`, arXiv:2310.18547):** Decomposes updates via skinny QR + `(u x u)` SVD with balanced `sqrt(Sigma)` scaling into frozen-base `(arena_b, arena_a)` slabs supporting `O(1)` rollback and per-row `slot_routing`.
4. **Post-quantization division-free certificate (Q3):** Computes `FixedRadixCertificate` (`frobenius_sq_scaled`, `spectral_norm_sq_scaled`, `leakage_energy_scaled`, `max_safety_dot_sq_scaled`) on the *post-quantized* `BF16` factors `(B_q, A_q)` and checks bounds via integer cross-multiplication.

## Evidence

- `[verified]` **Algebraic equivalence to dense BetaEdit Eq. (9):** Validated on synthetic `float64` matrices (`d_in=48, d_out=36, u=4`) with intermediate history updates between refreshes; relative Frobenius discrepancy between the `m x m` Cholesky reduction and the full `d_in x d_in` inverse is below `1e-12`.
- `[verified]` **Matrix-free rank-`r` Eckart-Young optimality & sequential leakage reduction (synthetic):** Skinny-QR truncation matches dense SVD truncation below `1e-11`; across 80 sequential unit-key `BF16` edits (`tau=20, lambda_1=200`), `BetaEditNullspaceProjector` achieved strictly lower anchor knowledge leakage (`||(W_0 + Delta_cum) K_0 - V_0||_F`) than static-projector `AlphaEdit-H` (`lambda_1=0`) while leaving `W_0` bit-identical.
- `[source claim]` **Long-horizon sequential editing collapse in `AlphaEdit` (`BetaEdit`, Liu et al., arXiv:2605.09285):** Because real LLM anchor covariances `K_0 K_0^T` are full-rank, singular-value thresholding (`< 0.02`) yields `P K_0 != 0`; `AlphaEdit` drops to `0.00%` efficacy at `T=5,000` CounterFact edits on `LLaMA3-8B` and `Qwen3-4B`, whereas `BetaEdit` retains `89.0%` and `90.3%` efficacy (`tau=1,000`).
- `[proposed]` **Real open-weights checkpoint transfer (Q1/Q4):** Evaluating `BetaEditNullspaceProjector` on actual `Qwen` / `Llama` MLP `down_proj` and untied `lm_head` activations—including measuring how `threshold_mode` (`absolute` vs `relative_max`) behaves under unnormalized activation norms—has not yet been executed in this repo.

## Result

The prototype resolves three structural traps ahead of GPU checkpoint runs: (1) eliminating `d_in x d_in` non-symmetric matrix inversions via exact `m x m` SPD Cholesky reduction, (2) preventing `AlphaEdit` pseudo-null-space drift across sequential multi-adapter mounts via `C^(t)` and periodic `tau`-refresh, and (3) certifying post-quantization `BF16` factors rather than pre-quantization `float64` intermediates.

## What would disprove this

1. **Real-checkpoint spectrum saturation (Q1/Q4):** On a real open-weights checkpoint (`Qwen3` or `Llama-3`), `K_0 K_0^T` has a flat singular spectrum where any `nullspace_threshold` preserving MMLU/perplexity leaves `m` too small to fit target edits, or `BetaEdit` performs no better than an unprojected ridge baseline at matched Frobenius norm.
2. **Same-subject / non-subject-token collision:** As noted in `arXiv:2605.09285` Section 7, edits anchored on shared function words or repeated subjects violate `⟨Delta_i, Delta_j⟩_F >= 0` assumptions and overwrite prior slots despite `C^(t)` regularization.

## Status

`open (prototype staged)` — reference PyTorch implementation and synthetic mathematical checks staged; Q1, Q3, and Q4 remain open pending real open-weights checkpoint execution.

## Next step

Run `docs/research/prototypes/2026-11-17-betaedit-projector.py` on an open-weights checkpoint (`Qwen3-0.6B` / `Qwen3.5-0.8B` or `Llama-3.2-1B`) comparing static `AlphaEdit`, `BetaEdit` (`tau=100`), and norm-matched random directions across `N=1..500` sequential edits on held-out perplexity and recall.
