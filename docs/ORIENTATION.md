# Orientation

Read this after `START-HERE.md`. It gives the shape of the research and the
rules for adding to it. It does not repeat the captures. The index in
`docs/research/INDEX.md` points to those.

## Mission

MDE studies one question: what changes when a model works in weight space
instead of token space? W-MCP is the concrete proposal. It maps the tools,
resources and handshake of MCP onto weight matrices, residual directions and
low-rank adapters. The promise is smaller context, direct capability mounts
and surgical edits. The cost is that an adapter becomes capability the host
cannot read.

The ethos repo decided this track is research, not roadmap. MDE is where it
stays alive. The serving substrate here is MEP. It is not a weight-space
runtime and it does not become one by a document change.

## What the captures established

- MCP has moved to a stateless wire. Version and client info live in `_meta`. Extensions are vendor-namespaced and standard hosts ignore unknown ones. This is verified against the published spec material.
- AlphaEdit projects edits into the null space of preserved knowledge keys, a space of directions the edit must leave alone. The published result is real, with a defined scope: declarative facts in MLP layers, not reasoning procedures or RL policies.
- The null-space idea transfers to a toy compositor. The corrected invariant, `delta_w @ K0.T`, measures about 3.7e-15. The prototype's own check prints 6.93 because it measures `K0 @ delta_w` instead.
- Weights have no shared coordinate system across model families. Neuron order, scale and rotation are arbitrary. A portable delta needs either a pinned checkpoint and basis or a shared semantic basis. No such wire exists.
- Attribution by spectral energy is a share, not a conservation law. It is defensible under sequential null-space projection. Outside that pipeline it double counts overlapping contributors.
- A null-space certificate is only meaningful if the host projects at load time. No shipped serving stack does this today.
- An ethos in weights is an ethos without a gate. The owner can read, diff and reject a text profile. An opaque delta is capability inside the forward pass.

## Hard vetoes

The convergence proposal calls the text layer Plane 1 and the compiled weight
layer Plane 2. These vetoes come from that proposal and hold until a human
changes them.

1. No opaque deltas. Every weight artifact must be re-derivable from published inputs with a public, boring compiler.
2. No automatic resolution of contested preferences, in any layer, including royalty-weighted merges. Conflicts go to the owner.
3. No closed-weight injection. Plane 2 exists only where the runtime can enforce projection, be turned off, and be audited.
4. No protocol proposal before external users have exercised the extension.
5. Plane 1 is the only place a person lives. Models, markets and agents are guests.

## The two architectures in the sketch

Target as server: a frontier model acts as client and a target network exposes
weights and forward probes as a server. The workflow is read, diagnose, write,
with atomic rollback. This is the debugging path.

Context as delta: external servers compile knowledge or policy into ephemeral
low-rank deltas instead of dumping tokens into the prompt. Three servers
become three composed deltas. This is the context-economy path.

Both need the same three things before they can ship: a shared basis, a way to
keep concurrent writes from colliding, and a verifier at the boundary to the
GPU. All three are open.

## Evidence bar

Each claim carries one of these labels:

- `verified`: checked against a primary source, or measured here with a command you can rerun.
- `source claim`: stated in a captured session, not checked by this repo.
- `proposed`: a new idea from the author. It has no standing until evidence lands.

A result without its command, inputs and environment is not `verified`.
Benchmark numbers from a source session stay `source claim` until reproduced.

## Working rules

- Read the processed note before the raw transcript. Raw files are third-party output preserved unedited. They contain plausible placeholders and unverified numbers on purpose.
- Do not rewrite a capture. Add a new dated document that says what changed.
- Claim a question before you work on it. Put your name or model and the date in `docs/OPEN-QUESTIONS.md`.
- Put results in a new dated file under `docs/research/`. Use the sections Question, Method, Evidence, Result, Status, Next step.
- Say what would disprove the result. Record a failure as a result with a status, not a silent omission.
- Keep personal data, secrets, credentials and machine paths out of the repo.
- The prose gate does not apply to quoted third-party output. It applies to everything you write.

## Definition of done

A question is closed when the answer is written down, the evidence is
rerunnable, the status is set, the changelog has a dated entry, the integrity
manifest is regenerated, the checks pass and the PR is reviewed. A negative
result closes a question too. An experiment that never ran does not.
