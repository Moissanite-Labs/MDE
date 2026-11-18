# Reference-only prototype. NOT MEP code, NOT shipped, NOT part of the stdlib-only promise.
# Requires PyTorch (`torch`) at runtime. Do not import this from `mep/`.
# Author: Gemini (machine-authored research prototype for Q1, Q3, and Q4), 2026-11-18.
"""History-aware BetaEdit null-space projector, periodic eigenspace refresh, and SGMV slab arena.

References:
- BetaEdit: Null-Space Constrained Sequential Model Editing (Liu et al., arXiv:2605.09285)
- AlphaEdit: Null-Space Constrained Knowledge Editing for LLMs (Fang et al., arXiv:2410.02355)
- Punica: Multi-Tenant LoRA Serving (Chen et al., arXiv:2310.18547)

Design invariants:
1. Exact Block-Triangular SPD Subspace Reduction: Replaces the non-symmetric (d_in x d_in)
   inverse in BetaEdit Eq. (9) with an algebraically identical (m x m) SPD Cholesky solve
   inside U_small (m = dim(pseudo-nullspace) <= d_in).
2. Matrix-Free Eckart-Young Rank-r Factorization: Computes skinny QR + (u x u) core SVD in
   O(u^2 (d_out + d_in)) without materializing dense (d_out x d_in) matrices, using balanced
   sqrt(Sigma) scaling to preserve BF16 dynamic range.
3. Post-Quantization Fixed-Radix Certificate: Computes division-free integer bounds directly
   on quantized (B_q, A_q) factors so BF16 rounding is included prior to slot activation.
4. Immutable Base Weights + O(1) Slot Arena: W_0 stays read-only; commits and rollbacks flip
   slot flags in pre-allocated (arena_b, arena_a) slabs.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

ThresholdMode = Literal["absolute", "relative_max", "relative_mean"]


@dataclass(frozen=True)
class BetaEditConfig:
    d_in: int
    d_out: int
    max_rank: int = 16
    max_slots: int = 1024
    lambda_leakage: float = 1000.0
    lambda_ridge: float = 10.0
    nullspace_threshold: float = 2e-2
    threshold_mode: ThresholdMode = "absolute"
    min_nullspace_dim: int = 1
    refresh_period: int = 1000
    radix_scale_exp2: int = 24
    compute_dtype: Any = None
    arena_dtype: Any = None

    def __post_init__(self) -> None:
        if self.compute_dtype is None:
            object.__setattr__(self, "compute_dtype", torch.float64)
        if self.arena_dtype is None:
            object.__setattr__(self, "arena_dtype", torch.bfloat16)
        if self.d_in <= 0 or self.d_out <= 0 or self.max_rank <= 0 or self.max_slots <= 0:
            raise ValueError("Dimensions, max_rank, and max_slots must be positive.")
        if self.lambda_leakage < 0.0 or self.lambda_ridge <= 0.0 or self.nullspace_threshold <= 0.0:
            raise ValueError("Invalid regularization or threshold parameters.")
        if not (1 <= self.min_nullspace_dim <= self.d_in):
            raise ValueError("min_nullspace_dim must lie in [1, d_in].")
        if self.refresh_period <= 0 or not (8 <= self.radix_scale_exp2 <= 48):
            raise ValueError("Invalid refresh_period or radix_scale_exp2.")


@dataclass(frozen=True)
class FixedRadixCertificate:
    """Division-free integer witness computed on post-quantization (B_q, A_q) factors."""

    radix_scale_exp2: int
    frobenius_sq_scaled: int
    spectral_norm_sq_scaled: int
    leakage_energy_scaled: int
    max_safety_dot_sq_scaled: int
    projector_epoch: int

    def verify_division_free(
        self,
        *,
        max_frob_sq_num: int,
        max_frob_sq_den: int = 1,
        max_leakage_num: Optional[int] = None,
        max_leakage_den: int = 1,
        max_safety_dot_sq_num: Optional[int] = None,
        max_safety_dot_sq_den: int = 1,
    ) -> bool:
        if max_frob_sq_den <= 0 or max_leakage_den <= 0 or max_safety_dot_sq_den <= 0:
            raise ValueError("Denominators must be strictly positive.")
        scale = 1 << self.radix_scale_exp2
        if self.frobenius_sq_scaled * max_frob_sq_den > max_frob_sq_num * scale:
            return False
        if max_leakage_num is not None and (
            self.leakage_energy_scaled * max_leakage_den > max_leakage_num * scale
        ):
            return False
        if max_safety_dot_sq_num is not None and (
            self.max_safety_dot_sq_scaled * max_safety_dot_sq_den > max_safety_dot_sq_num * scale
        ):
            return False
        return True


@dataclass(frozen=True)
class GatePolicy:
    max_frob_sq_num: int
    max_frob_sq_den: int = 1
    max_leakage_num: Optional[int] = None
    max_leakage_den: int = 1
    max_safety_dot_sq_num: Optional[int] = None
    max_safety_dot_sq_den: int = 1


@dataclass(frozen=True)
class StagedPatch:
    factor_b: torch.Tensor
    factor_a: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    truncation_rel_error: float
    certificate: FixedRadixCertificate


class BetaEditNullspaceProjector(nn.Module):
    """History-aware null-space constrained editor backed by an immutable SGMV slab arena."""

    w0: torch.Tensor
    cov_accum: torch.Tensor
    u_small: torch.Tensor
    cached_u_cov_u: torch.Tensor
    arena_b: torch.Tensor
    arena_a: torch.Tensor
    slot_active: torch.Tensor
    slot_ranks: torch.Tensor
    slot_generation: torch.Tensor

    def __init__(self, base_weight: torch.Tensor, config: BetaEditConfig) -> None:
        super().__init__()
        if base_weight.ndim != 2 or base_weight.shape != (config.d_out, config.d_in):
            raise ValueError(
                f"Expected base_weight of shape ({config.d_out}, {config.d_in}), got {tuple(base_weight.shape)}."
            )
        self.config = config
        dev = base_weight.device

        w0_buf = base_weight.detach().clone().to(device=dev, dtype=config.arena_dtype)
        w0_buf.requires_grad_(False)
        self.register_buffer("w0", w0_buf)
        self.register_buffer("cov_accum", torch.zeros((config.d_in, config.d_in), device=dev, dtype=config.compute_dtype))
        self.register_buffer("u_small", torch.eye(config.d_in, device=dev, dtype=config.compute_dtype))
        self.register_buffer("cached_u_cov_u", torch.zeros((config.d_in, config.d_in), device=dev, dtype=config.compute_dtype))
        self.register_buffer("arena_b", torch.zeros((config.max_slots, config.d_out, config.max_rank), device=dev, dtype=config.arena_dtype))
        self.register_buffer("arena_a", torch.zeros((config.max_slots, config.max_rank, config.d_in), device=dev, dtype=config.arena_dtype))
        self.register_buffer("slot_active", torch.zeros(config.max_slots, device=dev, dtype=torch.bool))
        self.register_buffer("slot_ranks", torch.zeros(config.max_slots, device=dev, dtype=torch.int32))
        self.register_buffer("slot_generation", torch.zeros(config.max_slots, device=dev, dtype=torch.int64))

        self._projector_epoch = 0
        self._commits_since_refresh = 0
        self._total_commits = 0
        self._slot_key_grams: dict[int, torch.Tensor] = {}
        self._safety_directions: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if f"{prefix}u_small" in state_dict:
            self.u_small = torch.empty_like(state_dict[f"{prefix}u_small"], device=self.w0.device)
        if f"{prefix}cached_u_cov_u" in state_dict:
            self.cached_u_cov_u = torch.empty_like(state_dict[f"{prefix}cached_u_cov_u"], device=self.w0.device)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def projector_epoch(self) -> int:
        return self._projector_epoch

    @property
    def commits_since_refresh(self) -> int:
        return self._commits_since_refresh

    @property
    def total_commits(self) -> int:
        return self._total_commits

    @property
    def nullspace_dim(self) -> int:
        return int(self.u_small.shape[1])

    @property
    def active_slot_count(self) -> int:
        return int(self.slot_active.sum().item())

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self.u_small @ self.u_small.T

    @torch.no_grad()
    def initialize_anchor_covariance(self, k0_or_gram: torch.Tensor, *, is_gram: bool = False) -> int:
        mat = k0_or_gram.to(device=self.cov_accum.device, dtype=self.config.compute_dtype)
        if is_gram:
            if mat.shape != (self.config.d_in, self.config.d_in):
                raise ValueError(f"Expected ({self.config.d_in}, {self.config.d_in}) Gram matrix.")
            gram = 0.5 * (mat + mat.T)
        else:
            if mat.ndim != 2 or mat.shape[0] != self.config.d_in:
                raise ValueError(f"Expected ({self.config.d_in}, n_0) anchor key matrix.")
            gram = mat @ mat.T
        self.cov_accum.copy_(self.config.lambda_leakage * gram)
        for slot_gram in self._slot_key_grams.values():
            self.cov_accum.add_(slot_gram)
        return self.refresh_projector()

    @torch.no_grad()
    def register_safety_direction(self, b_safe: torch.Tensor, a_safe: torch.Tensor) -> None:
        if b_safe.ndim != 2 or a_safe.ndim != 2 or b_safe.shape[0] != self.config.d_out or a_safe.shape[1] != self.config.d_in or b_safe.shape[1] != a_safe.shape[0]:
            raise ValueError("Incompatible safety factor shapes.")
        b64 = b_safe.detach().to(device=self.w0.device, dtype=self.config.compute_dtype)
        a64 = a_safe.detach().to(device=self.w0.device, dtype=self.config.compute_dtype)
        norm_sq = torch.sum((b64.T @ b64) * (a64 @ a64.T)).item()
        if norm_sq <= 0.0:
            raise ValueError("Safety direction must be non-zero.")
        scale = norm_sq ** -0.25
        self._safety_directions.append((b64 * scale, a64 * scale))

    @torch.no_grad()
    def refresh_projector(self, *, async_stream: Optional[Any] = None) -> int:
        def _recompute() -> int:
            eigvals, eigvecs = torch.linalg.eigh(0.5 * (self.cov_accum + self.cov_accum.T))
            eigvals = torch.clamp(eigvals, min=0.0)
            eps = self.config.nullspace_threshold
            if self.config.threshold_mode == "relative_max":
                m_ev = float(eigvals[-1].item())
                cutoff = eps * m_ev if m_ev > 0.0 else eps
            elif self.config.threshold_mode == "relative_mean":
                m_ev = float(eigvals.mean().item())
                cutoff = eps * m_ev if m_ev > 0.0 else eps
            else:
                cutoff = eps
            mask = eigvals < cutoff
            if int(mask.sum().item()) < self.config.min_nullspace_dim:
                u_sel = eigvecs[:, : self.config.min_nullspace_dim].contiguous()
                ev_sel = eigvals[: self.config.min_nullspace_dim]
            else:
                u_sel = eigvecs[:, mask].contiguous()
                ev_sel = eigvals[mask]
            self.u_small = u_sel
            self.cached_u_cov_u = torch.diag(ev_sel)
            self._projector_epoch += 1
            self._commits_since_refresh = 0
            return int(u_sel.shape[1])

        if async_stream is not None and self.cov_accum.is_cuda:
            async_stream.wait_stream(torch.cuda.current_stream(self.cov_accum.device))
            with torch.cuda.stream(async_stream):
                dim = _recompute()
            torch.cuda.current_stream(self.cov_accum.device).wait_stream(async_stream)
            return dim
        return _recompute()

    @torch.no_grad()
    def effective_matvec(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[0] != self.config.d_in:
            raise ValueError(f"Expected ({self.config.d_in}, u) keys.")
        k_comp = keys.to(device=self.w0.device, dtype=self.config.compute_dtype)
        out = self.w0.to(dtype=self.config.compute_dtype) @ k_comp
        active_idx = torch.nonzero(self.slot_active, as_tuple=False).flatten()
        if active_idx.numel() > 0:
            a_act = self.arena_a.index_select(0, active_idx).to(dtype=self.config.compute_dtype)
            b_act = self.arena_b.index_select(0, active_idx).to(dtype=self.config.compute_dtype)
            out = out + torch.matmul(b_act, torch.matmul(a_act, k_comp)).sum(dim=0)
        return out

    def forward(self, x: torch.Tensor, *, slot_routing: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] != self.config.d_in:
            raise ValueError(f"Expected trailing dimension {self.config.d_in}.")
        x_arena = x.to(device=self.w0.device, dtype=self.config.arena_dtype)
        base_out = torch.matmul(x_arena, self.w0.T)

        if slot_routing is not None:
            if x_arena.ndim < 2 or slot_routing.shape != (x_arena.shape[0],):
                raise ValueError("slot_routing must match batch dimension x.shape[0].")
            valid_mask = slot_routing >= 0
            if not torch.any(valid_mask):
                return base_out.to(dtype=x.dtype)
            routed_out = base_out.clone()
            rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
            slots = slot_routing.index_select(0, rows).to(dtype=torch.long)
            x_sel = x_arena.index_select(0, rows)
            a_sel = self.arena_a.index_select(0, slots)
            b_sel = self.arena_b.index_select(0, slots)
            mid = x_sel.shape[1:-1]
            x_flat = x_sel.reshape(x_sel.shape[0], -1, self.config.d_in)
            delta = torch.bmm(torch.bmm(x_flat, a_sel.transpose(1, 2)), b_sel.transpose(1, 2))
            routed_out.index_add_(0, rows, delta.reshape(x_sel.shape[0], *mid, self.config.d_out))
            return routed_out.to(dtype=x.dtype)

        active_idx = torch.nonzero(self.slot_active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            return base_out.to(dtype=x.dtype)
        a_act = self.arena_a.index_select(0, active_idx)
        b_act = self.arena_b.index_select(0, active_idx)
        s_cnt = active_idx.numel()
        a_cat = a_act.reshape(s_cnt * self.config.max_rank, self.config.d_in)
        b_cat = b_act.permute(1, 0, 2).reshape(self.config.d_out, s_cnt * self.config.max_rank)
        delta_flat = (x_arena.reshape(-1, self.config.d_in) @ a_cat.T) @ b_cat.T
        return (base_out + delta_flat.reshape_as(base_out)).to(dtype=x.dtype)

    @staticmethod
    def decompose_rank_r(
        left: torch.Tensor,
        right: torch.Tensor,
        max_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("Incompatible factor shapes.")
        q_l, r_l = torch.linalg.qr(left, mode="reduced")
        q_r, r_r = torch.linalg.qr(right.T, mode="reduced")
        u_c, s_vals, vh_c = torch.linalg.svd(r_l @ r_r.T, full_matrices=False)
        tot = float(torch.sum(s_vals * s_vals).item())
        if tot == 0.0:
            return (
                torch.zeros((left.shape[0], 1), device=left.device, dtype=left.dtype),
                torch.zeros((1, right.shape[1]), device=right.device, dtype=right.dtype),
                torch.zeros(1, device=left.device, dtype=left.dtype),
                0.0,
            )
        r_eff = min(max_rank, int(s_vals.numel()))
        s_kept = s_vals[:r_eff]
        disc = float(torch.sum(s_vals[r_eff:] * s_vals[r_eff:]).item())
        sqrt_s = torch.sqrt(s_kept)
        return (
            (q_l @ u_c[:, :r_eff]) * sqrt_s.unsqueeze(0),
            sqrt_s.unsqueeze(1) * (vh_c[:r_eff, :] @ q_r.T),
            s_kept,
            math.sqrt(max(0.0, disc) / tot),
        )

    @torch.no_grad()
    def _build_certificate(self, b_quant: torch.Tensor, a_quant: torch.Tensor) -> FixedRadixCertificate:
        b64 = b_quant.to(dtype=self.config.compute_dtype)
        a64 = a_quant.to(dtype=self.config.compute_dtype)
        _, r_b = torch.linalg.qr(b64, mode="reduced")
        _, r_a = torch.linalg.qr(a64.T, mode="reduced")
        s_q = torch.linalg.svdvals(r_b @ r_a.T)
        frob_sq = float(torch.sum(s_q * s_q).item())
        spec_sq = float((s_q[0] * s_q[0]).item()) if s_q.numel() > 0 else 0.0
        bt_b = b64.T @ b64
        ac_at = (a64 @ self.cov_accum) @ a64.T
        leak = max(0.0, float(torch.sum(bt_b * ac_at.T).item()))
        max_dot_sq = 0.0
        for b_s, a_s in self._safety_directions:
            d = float(torch.sum((b_s.T @ b64) * (a_s @ a64.T)).item())
            max_dot_sq = max(max_dot_sq, d * d)
        scale = float(1 << self.config.radix_scale_exp2)
        return FixedRadixCertificate(
            radix_scale_exp2=self.config.radix_scale_exp2,
            frobenius_sq_scaled=math.ceil(frob_sq * scale),
            spectral_norm_sq_scaled=math.ceil(spec_sq * scale),
            leakage_energy_scaled=math.ceil(leak * scale),
            max_safety_dot_sq_scaled=math.ceil(max_dot_sq * scale),
            projector_epoch=self._projector_epoch,
        )

    @torch.no_grad()
    def propose_patch(
        self,
        keys: torch.Tensor,
        target_values: Optional[torch.Tensor] = None,
        *,
        residuals: Optional[torch.Tensor] = None,
        max_rank: Optional[int] = None,
    ) -> StagedPatch:
        if (target_values is None) == (residuals is None):
            raise ValueError("Provide exactly one of target_values or residuals.")
        if keys.ndim != 2 or keys.shape[0] != self.config.d_in:
            raise ValueError(f"Expected ({self.config.d_in}, u) keys.")
        rank_cap = self.config.max_rank if max_rank is None else min(max_rank, self.config.max_rank)
        if rank_cap <= 0:
            raise ValueError("max_rank must be positive.")

        k64 = keys.to(device=self.w0.device, dtype=self.config.compute_dtype)
        if residuals is not None:
            r64 = residuals.to(device=self.w0.device, dtype=self.config.compute_dtype)
        else:
            assert target_values is not None
            r64 = target_values.to(device=self.w0.device, dtype=self.config.compute_dtype) - self.effective_matvec(k64)
        if r64.shape != (self.config.d_out, k64.shape[1]):
            raise ValueError("Residual/target shape mismatch.")

        k_proj = self.u_small.T @ k64
        m_dim = self.u_small.shape[1]
        spd = (
            self.config.lambda_ridge * torch.eye(m_dim, device=k64.device, dtype=k64.dtype)
            + self.cached_u_cov_u
            + (k_proj @ k_proj.T)
        )
        spd = 0.5 * (spd + spd.T)
        chol, info = torch.linalg.cholesky_ex(spd)
        solved = torch.cholesky_solve(k_proj, chol) if int(info.item()) == 0 else torch.linalg.solve(spd, k_proj)

        right64 = (self.u_small @ solved).T
        b64, a64, s_vals, trunc_err = self.decompose_rank_r(r64, right64, rank_cap)
        b_q, a_q = b64.to(dtype=self.config.arena_dtype), a64.to(dtype=self.config.arena_dtype)
        return StagedPatch(b_q, a_q, s_vals, int(b_q.shape[1]), trunc_err, self._build_certificate(b_q, a_q))

    @torch.no_grad()
    def commit_patch(
        self,
        patch: StagedPatch,
        keys: torch.Tensor,
        *,
        gate_policy: Optional[GatePolicy] = None,
        slot_id: Optional[int] = None,
        async_refresh_stream: Optional[Any] = None,
    ) -> int:
        if gate_policy is not None and not patch.certificate.verify_division_free(
            max_frob_sq_num=gate_policy.max_frob_sq_num,
            max_frob_sq_den=gate_policy.max_frob_sq_den,
            max_leakage_num=gate_policy.max_leakage_num,
            max_leakage_den=gate_policy.max_leakage_den,
            max_safety_dot_sq_num=gate_policy.max_safety_dot_sq_num,
            max_safety_dot_sq_den=gate_policy.max_safety_dot_sq_den,
        ):
            raise PermissionError(f"TCB gate rejected patch certificate: {patch.certificate}")
        r_eff = patch.effective_rank
        if r_eff > self.config.max_rank:
            raise ValueError("Patch rank exceeds arena max_rank.")
        if slot_id is None:
            inactive = torch.nonzero(~self.slot_active, as_tuple=False).flatten()
            if inactive.numel() == 0:
                raise RuntimeError("All SGMV arena slots are occupied.")
            target = int(inactive[0].item())
        else:
            if not (0 <= slot_id < self.config.max_slots):
                raise IndexError("slot_id out of range.")
            if bool(self.slot_active[slot_id].item()):
                self.rollback_slot(slot_id, revert_covariance=True)
            target = slot_id

        self.arena_b[target].zero_()
        self.arena_a[target].zero_()
        self.arena_b[target, :, :r_eff].copy_(patch.factor_b.to(device=self.w0.device, dtype=self.config.arena_dtype))
        self.arena_a[target, :r_eff, :].copy_(patch.factor_a.to(device=self.w0.device, dtype=self.config.arena_dtype))
        self.slot_ranks[target] = r_eff
        self.slot_generation[target] += 1
        self.slot_active[target] = True

        k64 = keys.to(device=self.w0.device, dtype=self.config.compute_dtype)
        gram = k64 @ k64.T
        self.cov_accum.add_(gram)
        k_proj = self.u_small.T @ k64
        self.cached_u_cov_u.add_(k_proj @ k_proj.T)
        self._slot_key_grams[target] = gram
        self._commits_since_refresh += 1
        self._total_commits += 1
        if self._commits_since_refresh >= self.config.refresh_period:
            self.refresh_projector(async_stream=async_refresh_stream)
        return target

    @torch.no_grad()
    def rollback_slot(self, slot_id: int, *, revert_covariance: bool = True) -> None:
        if not (0 <= slot_id < self.config.max_slots):
            raise IndexError("slot_id out of range.")
        self.slot_active[slot_id] = False
        self.slot_ranks[slot_id] = 0
        self.arena_b[slot_id].zero_()
        self.arena_a[slot_id].zero_()
        self.slot_generation[slot_id] += 1
        gram = self._slot_key_grams.pop(slot_id, None)
        if revert_covariance and gram is not None:
            self.cov_accum.sub_(gram)
            torch.diagonal(self.cov_accum).clamp_(min=0.0)
            self.cached_u_cov_u.copy_(self.u_small.T @ self.cov_accum @ self.u_small)

    @torch.no_grad()
    def knowledge_leakage_norm(self, k0: torch.Tensor, v0: Optional[torch.Tensor] = None) -> float:
        k64 = k0.to(device=self.w0.device, dtype=self.config.compute_dtype)
        if v0 is None:
            active_idx = torch.nonzero(self.slot_active, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                return 0.0
            a_act = self.arena_a.index_select(0, active_idx).to(dtype=self.config.compute_dtype)
            b_act = self.arena_b.index_select(0, active_idx).to(dtype=self.config.compute_dtype)
            drift = torch.matmul(b_act, torch.matmul(a_act, k64)).sum(dim=0)
            return float(torch.linalg.matrix_norm(drift, ord="fro").item())
        v64 = v0.to(device=self.w0.device, dtype=self.config.compute_dtype)
        return float(torch.linalg.matrix_norm(self.effective_matvec(k64) - v64, ord="fro").item())
