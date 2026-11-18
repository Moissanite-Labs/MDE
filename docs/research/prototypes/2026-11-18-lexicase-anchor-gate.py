# Toy demonstration of the Two-Tiered Lexicase Anchor Gate translated to
# low-rank weight deltas (proposed, 2026-11-18). Mirrors the shape of
# prototypes/2026-11-14-taste-compositor-fixcheck.py.
#
# Setup. Base model W0: 16 inputs -> 4 logits. Four (x, expected_y) anchor
# pairs are stored in the delta header as behavioral canaries. Anchors are
# sampled near decision boundaries, like the Triangular Sum ladder in the
# spliceware proposal: a spread of gaps, not a comfortable cluster.
#
# What it shows. A rank-4 delta built to flip all four anchors passes a
# generous Frobenius norm budget, and is caught by both the anchor gate and
# the null-space overlap check. An honest delta built inside null(K0) passes
# everything. A weaker attack that flips exactly one anchor is refused by
# Tier 1 (4/4 required) and admitted to Tier 2 (>= 3/4). Nothing here touches
# a real checkpoint; the question this informs is Q3, with Q1 as context.
import numpy as np

rng = np.random.default_rng(20261118)

D_IN, D_OUT = 16, 4
W0 = rng.standard_normal((D_OUT, D_IN)) * 2.0
NORM_BUDGET = 0.5 * np.linalg.norm(W0, "fro")

# Preserved keys K0 (facts the delta must not move): 12 keys in 16 dims,
# rank 12, null(K0) is 4 dimensional.
K0 = rng.standard_normal((12, D_IN))
_, _, Vt = np.linalg.svd(K0, full_matrices=True)
rank = np.linalg.matrix_rank(K0)
N = Vt.T[:, rank:]  # 16 x 4 null basis

# Anchors sampled near decision boundaries.
cand = rng.standard_normal((4000, D_IN))
logits = cand @ W0.T
sorted_logits = np.sort(logits, axis=1)
gaps = sorted_logits[:, -1] - sorted_logits[:, -2]
# Anchors on a difficulty ladder: easiest and hardest rungs from the same
# sample, not four identical near-ties.
order = np.argsort(gaps)
n = len(order)
anchors_x = cand[order[[int(0.01 * n), int(0.10 * n), int(0.30 * n), int(0.60 * n)]]]
anchors_y = np.argmax(anchors_x @ W0.T, axis=1)
logits0 = anchors_x @ W0.T
gaps_anchors = np.sort(logits0, axis=1)[:, -1] - np.sort(logits0, axis=1)[:, -2]

def eval_delta(dW):
    fro = np.linalg.norm(dW, "fro")
    leak = np.linalg.norm(dW @ K0.T)
    pred = np.argmax(anchors_x @ (W0 + dW).T, axis=1)
    hits = int(np.sum(pred == anchors_y))
    return fro, leak, hits

def verdict(name, dW):
    fro, leak, hits = eval_delta(dW)
    print(f"{name}")
    print(f"  ||dW||_F        = {fro:.3e}   budget {NORM_BUDGET:.3f}   {'PASS' if fro <= NORM_BUDGET else 'FAIL'}")
    print(f"  ||dW @ K0.T||_F = {leak:.3e}   (must be ~0)")
    print(f"  anchors         = {hits}/4   Tier1 {'PASS' if hits == 4 else 'FAIL'}   Tier2 {'PASS' if hits >= 3 else 'FAIL'}\n")
    return fro, leak, hits

print(f"||W0||_F = {np.linalg.norm(W0, 'fro'):.3f}   null(K0) dim = {D_IN - rank}   base anchors: {np.sum(np.argmax(anchors_x @ W0.T, axis=1) == anchors_y)}/4\n")

# A. Targeted flip attack: boost every anchor's runner-up class by rho.
#    dW @ anchors_x.T = rho * F, so each anchor's runner-up logit rises by rho.
F = np.zeros((D_OUT, 4))
logits0 = anchors_x @ W0.T  # 4 x 4
for i in range(4):
    order = np.argsort(-logits0[i])
    F[order[1], i] = 1.0  # runner-up class gets the boost
attack_dirs = F @ np.linalg.pinv(anchors_x.T)  # 4 x 16, rank <= 4
flips = []
for rho in np.linspace(0.1, 5.0, 100):
    dW = rho * attack_dirs
    _, _, hits = eval_delta(dW)
    flips.append((rho, hits))
rho_all = next((r for r, h in flips if h == 0), None)
if rho_all is not None:
    verdict("A. targeted rank-4 flip attack (0/4 anchors survive)", rho_all * attack_dirs)
else:
    # No rho in the sweep flips all four; report the strongest observed.
    rho_best = min(flips, key=lambda t: t[1])[0]
    verdict("A. targeted rank-4 flip attack (best observed)", rho_best * attack_dirs)

# B1/B2. Honest deltas: random, inside null(K0), two sizes. The large one
# shows the failure mode plainly: key leakage stays ~0 while near-boundary
# anchor behavior degrades. A null-space certificate is not a behavior
# certificate.
dW_honest = rng.standard_normal((D_OUT, D_IN - rank)) @ N.T
dW_small = dW_honest * (0.05 * NORM_BUDGET / np.linalg.norm(dW_honest, "fro"))
verdict("B1. honest null-space delta, small norm", dW_small)
dW_large = dW_honest * (0.5 * NORM_BUDGET / np.linalg.norm(dW_honest, "fro"))
verdict("B2. honest null-space delta, large norm", dW_large)

# C. Stepping-stone delta: boost the runner-up class of exactly one anchor
# (the closest to its boundary) until exactly one anchor flips.
i_star = int(np.argmin(gaps_anchors))
F1 = np.zeros((D_OUT, 4))
order = np.argsort(-logits0[i_star])
F1[order[1], i_star] = 1.0
attack1_dirs = F1 @ np.linalg.pinv(anchors_x.T)
rho_one = next((r for r in np.linspace(0.05, 5.0, 200) if eval_delta(r * attack1_dirs)[2] == 3), None)
if rho_one is not None:
    verdict("C. stepping-stone delta (expect Tier1 FAIL, Tier2 PASS)", rho_one * attack1_dirs)
else:
    print("C. stepping-stone delta: no single-flip rho in the sweep; Tier 2 lane not demonstrated on this seed.\n")
