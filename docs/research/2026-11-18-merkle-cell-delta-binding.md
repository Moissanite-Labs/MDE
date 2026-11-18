# Content-addressed delta binding, distilled from spliceware (2026-11-18)

Processed note. Raw source: [2026-11-18-spliceware-sacred-splicer-raw.md](2026-11-18-spliceware-sacred-splicer-raw.md) (Turns 7-9, 13-22). Captured 2026-11-18 from `Moissanite-Labs/spliceware` at `7a1a27fec1c9ad886aaac0468846b4a6965e20de`. Mapped to **Q2** (what binds a delta to a checkpoint?) and **Q6** (what does the text-space lineage ledger look like?).

## Question

Can a delta carry its own binding to the checkpoint it was built against, together with enough lineage to replay and audit the transition, without a central coordinator?

## Method

Architectural analysis of the spliceware corpus's `Cell` envelope and the session's determinism stack (Turns 7-9), followed by a mapping onto Plane 1 and Plane 2 of the convergence proposal. Reference points: [2026-11-14-alphaedit-and-scwsp.md](2026-11-14-alphaedit-and-scwsp.md), [2026-11-14-marketplace-provenance.md](2026-11-14-marketplace-provenance.md).

## Evidence

`source claim`:

- The corpus collapses the runtime to one 1,088-byte `Cell`: 32-byte `left` (Merkle link to `ancestor_a`), 32-byte `right` (Merkle link to `ancestor_b`), 1,024-byte `body`. One 256-bit ARX sponge hashes preimage, address, and wire bytes identically (`RAM == WIRE == HASH`).
- A cell travels in one unfragmented UDP datagram and is lazily fetched on arena cache miss in `resolve()`. No packet carries control-plane metadata; the bytes are the genome.
- Determinism primitives used: RFC 6979 / Ed25519 for signatures, AES-SIV / deterministic AEAD for deduplicable payloads, VRFs for verifiable tournament entropy, BIP-32 for capability subkeys, Merkle-CRDTs for causal ordering without consensus.

`verified`:

- The arithmetic is self-consistent on inspection: 1,088 bytes < 1,472 MTU; the ARX sponge is invertible-free and deterministic across platforms for integer ops; `code-13` shows the in-cell prior vector updated multiplicatively with saturating `[2, 255]` bounds, which is replayable given identical parent hashes and tick.

`proposed`:

- Bind a W-MCP delta the way a `Cell` binds its parents: a 64-byte dual-parent envelope where `left[32]` is the base checkpoint digest and `right[32]` is the delta/donor digest, hashed by a deterministic sponge. A host that only has the envelope can verify both ends before any FLOP of the delta touches weights.
- Map Plane 2 weight transitions onto a Plane 1 human-auditable lineage ledger: every accepted transition appends `(parent_checkpoint_hash, delta_hash, gate_verdict, tick)` to a signed record. Plane 1 stays the only place a person lives; the ledger is text, the deltas are weights.
- Derive crossover/projection seeds deterministically from `H(parent_a || parent_b || tick)` so any peer replays the lineage bit-for-bit without coordinating randomness.

## Result

The `RAM == WIRE == HASH` envelope answers the transport half of Q2. It does not answer the semantic half: a digest binds bytes, not meaning. Two checkpoints from different families can both match digests while meaning nothing to each other; the gauge problem is untouched by hashing. The binding that matters for Plane 1 is a human-readable attestation chain (who built this delta, from which base, what gate verdict it passed), anchored to the digests by signature. The corpus demonstrates the anchoring pattern, not the attestation policy.

For Q6, the direct answer is: a signed append-only lineage graph whose nodes are checkpoint/delta pairs and whose edges are gated transitions, stored as text, with the Plane 2 payloads referenced by hash. Whether it belongs in MDE or upstream in ethos is an open placement question, not a research one.

## Status

Open (Q2 and Q6 stay open). Q2 needs a runnable mock binding digest, geometry, quantization, and a behavioral target, with correct acceptance and rejection cases. Q6 needs a design note checked against the ethos export bundle format.

Falsification criteria:

- Disproved if float16/bfloat16 GPU kernel non-determinism across hardware vendors breaks bit-exact activation and output anchor replay even when parent checkpoint and delta hashes match exactly (tolerance intervals would be required instead of exact byte equality, which weakens the envelope's audit story).
- Disproved for the Plane 1 ledger if signed lineage records cannot be verified by a third party without the original build environment.

## Next step

Q2: specify the minimum handshake fields (checkpoint digest, layer geometry, quantization, basis identifiers or behavioral target) and build the mock accept/reject harness. Q6: draft the design note against the ethos bundle format and decide placement (MDE or ethos).
