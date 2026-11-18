# Open questions

Numbered tasks for research models. Claim one by adding your name or model and
the date to its claim line before you start. Write results to a new dated file
in `docs/research/` using the sections in `START-HERE.md`.

Priority: P0 blocks a weight-space design, P1 advances it, P2 is adjacent work
that can be done now.

## Q1 (P0) Does null-space projection hold on a real checkpoint?

Background: the prototype guards a random 32 by 128 matrix and measures the
corrected invariant at about 3.7e-15. AlphaEdit builds its protected space from
cached keys of facts that must not change. A random matrix is not a protected
knowledge set.

Task: build the protected space from a real open-weights checkpoint with the
AlphaEdit construction. Measure the invariant after a null-space-projected
edit. Report the rank of the protected space, how coverage of preserved
behaviors was chosen, and the largest leakage you observe.

Closes when: the corrected invariant is measured on a named checkpoint with a
rerunnable command, or a failing measurement shows where the construction
breaks.

Disproves: leakage above the stated tolerance on the target checkpoint.

Status: open. Claim: none.

## Q2 (P0) What binds a delta to a checkpoint?

Background: neuron order, scale and rotation are arbitrary across model
families. A raw delta from one model does not mean the same thing in another.
The W-MCP sketch says the handshake must pin a checksum and a basis but defines
neither.

Task: specify the minimum fields a weight-space handshake needs before it
accepts a delta: checkpoint digest, layer geometry, quantization, and basis
identifiers or a behavioral target. Show a transfer test between two
checkpoints from different families that either works or is correctly
rejected.

Closes when: a written schema and a runnable mock show correct acceptance and
correct rejection.

Disproves: a delta that applies cleanly to a checkpoint whose digest does not
match the one it was built for.

Status: in review. Claim: Gemini (machine-authored), 2026-11-17 — see docs/research/2026-11-17-q2-dawe-gpatch-splice.md.

## Q3 (P0) Can a cheap gate catch a malicious delta?

Background: a delta is an opaque binary patch. A small change can ablate a
refusal direction. Norm bounds and inner products against protected subspaces
are cheap linear algebra, but no shipped runtime applies them at load time.

Task: write a threat model for an untrusted packet. Implement candidate
certificates such as a spectral norm cap, an overlap test against a protected
subspace, and a checkpoint digest check. Try to build a packet that passes the
certificates but changes a protected behavior. Report false positives and
false negatives.

Closes when: the gate is runnable and each attack attempt has a recorded
outcome.

Disproves: a packet that passes every certificate and silently disables a
protected behavior.

Status: open. Claim: none.

## Q4 (P1) How do concurrent adapters avoid overwriting each other?

Background: a small edit moved one model a long way on a benchmark and barely
moved a larger one. Multiple servers writing deltas at once can regress
unrelated skills.

Task: define an allocation rule that gives each incoming delta a subspace
orthogonal to capabilities already mounted. Compose N adapters on a shared
base and measure regression as N grows. Say where the rule breaks.

Closes when: the allocation rule and a regression curve are recorded with a
rerunnable command.

Disproves: measurable damage to a protected capability after an accepted
delta.

Status: open. Claim: none.

## Q5 (P1) Do the headline numbers reproduce?

Background: the sketch quotes 7.9 percent to 48.1 percent on MATH-500, and
8.35 percent to 37.80 percent on the same benchmark for a different edit.
These are source authors' results. This repo has not run them.

Task: reproduce both with public code and named checkpoints. Record the exact
commands, data, hardware and seeds. If reproduction fails, report the gap
instead of adjusting the claim.

Closes when: a reproduction report carries the numbers or an honest failure
with the setup.

Disproves: any use of these numbers as established facts in MDE documents.

Status: open. Claim: none.

## Q6 (P2) What does the text-space lineage ledger look like?

Background: the convergence proposal says lineage graphs and royalty shares
are cheap and buildable on ethos export bundles, with no weights involved. It
may belong upstream in ethos rather than here.

Task: design a signed lineage graph on export bundles: parent profiles,
operators applied at import, and a share of approved revisions per source.
Check it against the existing bundle format before writing code.

Closes when: a design note says what fits the current format, what needs a new
one, and where the work belongs.

Status: open. Claim: none.

## Out of scope here

- Republishing the SC-WSP brief with a named author. Its audit explains why not.
- Weight injection into closed-weight APIs.
- Treating numbers from a raw session as facts.
- Changing MEP v0.1 in this repo. Protocol changes go to ethos.
