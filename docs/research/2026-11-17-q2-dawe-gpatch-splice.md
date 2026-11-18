# Q2: Checkpoint-bound sparse deltas via `geodesic-patch/1` (`dawe_gpatch` splice)

Author: Gemini (machine-authored research note and Splicer Protocol graft), 2026-11-17.

## Question

**Q2 (P0) — What binds a delta to a checkpoint?** Specify the minimum fields for a weight-space handshake and delta envelope (checkpoint digest, layer geometry, quantization/dtype, serving-template state, and derivation lineage) and demonstrate a deterministic transfer/tamper test without breaking `MEP v0.1`'s stdlib-only contract.

## Method

Following the Splicer Protocol (`v0.1.0`) layout (`vendor/splices/dawe_gpatch/`), we extracted the pure-stdlib `geodesic-patch/1` (`.gpatch`) Safetensors sparse-delta core (`slice.py`) from [`bandr-ai/dawe`](https://github.com/bandr-ai/dawe) (`dawe/patch.py`, Apache-2.0), severed `dawe.telemetry` and CLI globals, and bridged it onto `Moissanite-Labs/MDE` (`mep`) via a single host adapter (`seam.py`) and contract test suite (`tests/test_splice_dawe_gpatch.py`) with zero changes to `mep/`.

To align with `docs/ORIENTATION.md` Hard Veto #1 (*no opaque deltas*), `seam.py` stamps `source.lineage` with the active `Ethos` document SHA-256 map plus an explicit `derivation` recipe slot, and enforces an optional `require_derivation=True` verification gate alongside full checkpoint and side-file SHA-256 verification.

## Evidence

- `[verified]` **Four-layer exact checkpoint binding (`vendor/splices/dawe_gpatch/slice.py`, `tests/test_splice_dawe_gpatch.py`):** A `.gpatch` manifest binds a sparse weight delta (`index/<t>`, `old/<t>`, `new/<t>` in valid Safetensors framing) to an exact target via:
  1. `source.layout_sha256`: SHA-256 over sorted `name \t dtype \t shape` lines across all `*.safetensors` shards.
  2. `source.fingerprint` and `result.fingerprint` (plus per-tensor `source_sha256` / `result_sha256` and `payload_sha256`): SHA-256 over exact raw tensor byte spans, enabling verified `apply` and byte-for-byte `restore` (`test_roundtrip_apply_restore_and_derivation_gate`).
  3. `source.config` and `source.tokenizer`: SHA-256 map over `config.json`, `generation_config.json`, `tokenizer.json`, `chat_template.jinja`, and vocabulary files. Mutating `chat_template.jinja` after patch creation is deterministically rejected with `incompatible_checkpoint` (`test_template_drift_and_opaque_delta_rejection`).
  4. `source.lineage`: `Ethos` document digests plus `derivation` compiler metadata; patches lacking `derivation` fail `wmcp_patch_verify(..., require_derivation=True)` (`test_template_drift_and_opaque_delta_rejection`).
- `[verified]` **Host isolation and confinement:** All paths pass through `mep.integrity.confined_path` (`test_confinement_rejects_escape_paths`), `mep/` (`MEP v0.1`) has zero modified lines, and `dependencies = []` remains intact.
- `[source claim]` **Real-checkpoint readout repair (`bandr-ai/dawe` write-up on `Qwen/Qwen3.5-4B`):** Fitting closed-form ridge regression vectors on final-layer hidden states to shift pre-softmax logits edited 1-2 rows (`2,557` to `5,064` BF16 weights; `23.6 KB` to `43.7 KB` `.gpatch`), lifting held-out BFCL XML accuracy from `73.5%` to `85.8%` and plan-first accuracy from `20.1%` to `88.2%` while keeping JSON `v1` at `87.7%` (single 204-item custom matcher run; not re-executed in this repo).
- `[proposed]` **Untied low-rank SGMV runtime & history-aware null-space solver (for Q1/Q3/Q4):** Because `Qwen3.5-4B` ties `embed_tokens.weight` between input lookup and output readout, row edits in `embed_tokens.weight` also perturb input token embeddings. Mounting `k`-row readout patches as untied rank-`k` `(B, A)` slots in a frozen-base SGMV arena—solved inside a history-aware `BetaEdit` null space (`C^(t) = lambda_1 K_0 K_0^T + sum_{i<t} K_i K_i^T` reduced via block-triangular SPD Cholesky inside `U_small`) and gated by post-quantization fixed-radix norm/leakage certificates—would decouple readout from input embeddings and prevent sequential multi-patch interference.

## Result

For **same-checkpoint weight-space transport**, the minimum sound binding tuple is `(layout_sha256, tensor_fingerprint, config_sha256_map, tokenizer_and_chat_template_sha256_map, payload_sha256, derivation_lineage)`. Omitting `tokenizer` / `chat_template.jinja` hashes leaves a critical soundness hole because logit-readout and steering deltas depend directly on prompt framing and special-token IDs. Exact byte binding intentionally refuses cross-checkpoint or cross-quantization transfer, isolating gauge/basis translation cleanly to Q1 and Q4.

## What would disprove this

1. **False-accept soundness failure:** Constructing two checkpoints that differ in any weight byte, tensor shape/dtype, `config.json`, `tokenizer.json`, or `chat_template.jinja` while passing `wmcp_patch_verify(..., full=True)` (barring SHA-256 collision).
2. **Sufficiency gap in serving state:** Showing an unpinned serving parameter outside `CONFIG_FILES` and `TOKENIZER_FILES` (for example, external RoPE scaling overrides, quantization runtime flags, or custom tool-parser regexes) that silently invalidates a verified `.gpatch` without triggering `check_checkpoint`.
3. **Opaque-delta bypass:** Demonstrating that metadata-only `source.lineage.derivation` suffices for Hard Veto #1 without re-running the deterministic compiler to reproduce `new/<tensor>` bytes from published calibration states.

## Status

`in review` — byte-level and serving-template binding (`Q2` same-checkpoint regime) implemented and verified in `vendor/splices/dawe_gpatch/` and `tests/test_splice_dawe_gpatch.py`; cross-seed/cross-quantization basis handshake and deterministic replay of `derivation` remain open.

## Next step

1. Extend `CONFIG_FILES` / handshake `_meta` to pin runtime execution flags (quantization kernel mode, RoPE override, tool parser ID) identified in Disproves #2.
2. Pair `.gpatch` manifests with a deterministic replay verifier for `source.lineage.derivation` (Hard Veto #1) and benchmark untied rank-`k` readout projection on an open checkpoint under Q1/Q4.
