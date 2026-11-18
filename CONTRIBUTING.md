# Contributing

A contribution here is a claim you can check. This page says how work lands
in MDE.

## What this repo is

The W-MCP research line: captures inherited from ethos, the analysis written
around them, and the questions still open. The MEP server (`mep/`, `v0.1/`,
`template/`, `tests/`) is the serving substrate. Keep it working. Do not add
weight-space behavior to it as if it were MEP v0.1.

## Ground rules

- Plain language. Short sentences. Explain a technical term in ordinary words the first time you use it.
- Date every document and every change. Put the reason in `CHANGELOG.md`.
- Label claims. Use `verified` only for something checked against a primary source or measured here with a rerunnable command. Use `source claim` for something a captured session asserted. Use `proposed` for a new idea.
- Put the evidence next to the claim: a source link, a command, a measurement, or a test. A number without its setup is not evidence.
- Keep raw captures verbatim in `*-raw.md` files. Write new analysis in a new dated file. Never edit a capture to make it agree with a later result.
- No personal data, secrets, credentials, model keys or machine paths.
- No new Python dependency in the substrate. The reference server stays standard library only.
- No CI yet. Run the three checks from `README.md` before sending a PR.
- One branch and one PR per change. Squash with a clean message. No force push.
- Give one independent review before merging code. Research notes can be merged by their author once checked.

## How to take a research question

1. Add your name or model and the date to the claim line for that question in `docs/OPEN-QUESTIONS.md`.
2. Read the processed note listed on the question, then the raw source if you need it.
3. Write the result to `docs/research/<date>-<question-id>-<short-name>.md` with the sections Question, Method, Evidence, Result, Status, Next step.
4. State what would disprove your result. If the experiment did not run, say so.
5. Update `CHANGELOG.md` and regenerate `SHA256SUMS` with the integrity tool.

## Protocol changes

The protocol lives in ethos. If a result should change MEP, write the finding
here first and open an issue on ethos with the problem, not a patch. Vendor
extensions and protocol changes belong there, not in this repo.

## Credit

Contributions are Apache-2.0 by default. Record your name or handle in the PR
description if you want it in `NOTICE`.
