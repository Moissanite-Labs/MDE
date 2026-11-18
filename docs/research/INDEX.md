# Research index

Everything captured in this repo, with its status. Files came from ethos at
commit `afdb3a4` and are kept byte for byte. Read the processed note in a pair
before its raw source. A raw file is third-party output, not a claim of this
repo.

## W-MCP core

| File | Kind | Standing |
| --- | --- | --- |
| [2026-11-14-w-mcp-sketch.md](2026-11-14-w-mcp-sketch.md) | processed sketch | The research direction. Three bottlenecks are open. Not a design for MEP. |
| [2026-11-14-mcp-session-notes.md](2026-11-14-mcp-session-notes.md) | processed notes | Verified MCP background: stateless wire, SEPs, extensions, context cost. |
| [2026-11-14-mcp-session-raw.md](2026-11-14-mcp-session-raw.md) | raw transcript | Source for the notes and the sketch. Unedited, includes diagram markup. |

## AlphaEdit and SC-WSP

| File | Kind | Standing |
| --- | --- | --- |
| [2026-11-14-alphaedit-and-scwsp.md](2026-11-14-alphaedit-and-scwsp.md) | processed notes | AlphaEdit verified within its scope. SC-WSP is a design hypothesis with no benchmark. |
| [2026-11-14-alphaedit-and-scwsp-raw.md](2026-11-14-alphaedit-and-scwsp-raw.md) | raw transcript | Verbatim source for the notes above. |
| [2026-11-14-scwsp-brief.md](2026-11-14-scwsp-brief.md) | audit of a brief | Not adopted. The printed invariant has the wrong orientation. The footer verification claim is false for that product. |
| [2026-11-14-compositor-prototype.md](2026-11-14-compositor-prototype.md) | prototype analysis | Corrected invariant measures 3.7e-15 on a toy model. Reference only. |

## Marketplace and provenance

| File | Kind | Standing |
| --- | --- | --- |
| [2026-11-14-taste-marketplace.md](2026-11-14-taste-marketplace.md) | processed notes | The five-step pipeline and the weaving algebra use real methods. The product does not exist. Weight packets cannot be gated by the host. |
| [2026-11-14-taste-marketplace-raw.md](2026-11-14-taste-marketplace-raw.md) | raw transcript | Verbatim source for the notes above. |
| [2026-11-14-marketplace-provenance.md](2026-11-14-marketplace-provenance.md) | processed notes | Watermarking and lineage are real. The zero-knowledge framing is mislabeled. A deterministic check suffices. |
| [2026-11-14-marketplace-provenance-raw.md](2026-11-14-marketplace-provenance-raw.md) | raw transcript | Verbatim source for the notes above. |

## Cellular Substrates & Anchor-Gated Verification (from `spliceware`)

| File | Kind | Standing |
| --- | --- | --- |
| [2026-11-18-anchor-gated-load-verification.md](2026-11-18-anchor-gated-load-verification.md) | processed notes (Q3, informs Q1) | Anchor-gate mechanism analyzed on a toy. Norm gate waves a targeted rank-4 attack through; canary and null-space checks catch it. Real-checkpoint answer still open. |
| [2026-11-18-merkle-cell-delta-binding.md](2026-11-18-merkle-cell-delta-binding.md) | processed notes (Q2, Q6) | `RAM == WIRE == HASH` envelope answers transport binding. Semantic binding and ledger placement still open. |
| [2026-11-18-homologous-splice-and-adaptive-priors.md](2026-11-18-homologous-splice-and-adaptive-priors.md) | processed notes (Q4) | Seam-admission and A/B shadow evaluation are candidate composition rules. Nothing measured on real checkpoints yet. |
| [2026-11-18-spliceware-sacred-splicer-raw.md](2026-11-18-spliceware-sacred-splicer-raw.md) | raw transcript | Turns 7-9 and 13-22 verbatim plus the post-session proposals. Unedited. Contains unverified numbers on purpose. |
| [prototypes/2026-11-18-lexicase-anchor-gate.py](prototypes/2026-11-18-lexicase-anchor-gate.py) | prototype | Toy numpy demonstration of the anchor gate. Run it from the repo root. No real checkpoint. |

## Prototypes

| File | Standing |
| --- | --- |
| [prototypes/2026-11-14-scwsp-brief-generator.py](prototypes/2026-11-14-scwsp-brief-generator.py) | Preserved verbatim. Generates the audited brief. Needs `weasyprint` and `pypdf`. |
| [prototypes/2026-11-14-taste-compositor.py](prototypes/2026-11-14-taste-compositor.py) | Preserved verbatim. Runs on numpy. Its final check prints the wrong product. |
| [prototypes/2026-11-14-taste-compositor-fixcheck.py](prototypes/2026-11-14-taste-compositor-fixcheck.py) | Corrected check written during capture. Run it from the repo root. |

## Proposal

| File | Standing |
| --- | --- |
| [../proposal/2026-11-14-ethos-convergence.md](../proposal/2026-11-14-ethos-convergence.md) | A proposal from ethos. Two planes, one compiler, five hard vetoes. Not implemented. Read it as a position, not a spec. |
