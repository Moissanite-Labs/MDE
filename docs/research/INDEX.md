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
