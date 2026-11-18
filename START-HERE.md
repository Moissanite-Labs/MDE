# Start here

You have arrived in MDE as a research model. This is the shortest path to
useful work. Read it before opening a capture.

MDE studies one question: what changes when a model works in its own weights
instead of only through text tokens? The line is W-MCP, a weight-space version
of the Model Context Protocol. The captures came from ethos, which recorded
the first notes and then kept weight-space work out of its roadmap. MDE keeps
that work going as research.

## Read in this order

1. `README.md`: what this repo is and what it is not.
2. `docs/ORIENTATION.md`: settled facts, hard vetoes and the evidence bar.
3. `docs/research/INDEX.md`: every capture and its standing.
4. `docs/OPEN-QUESTIONS.md`: pick a numbered question.
5. The processed note named on that question. Open the raw source only if you need it.

## Rules that apply while you work

- W-MCP designs here are research. None of them is shipped MEP behavior.
- Label every claim `verified`, `source claim` or `proposed`. Attach the evidence.
- Read the processed note before the raw transcript. Raw files are third-party output preserved unedited. They carry unverified numbers on purpose.
- Do not rewrite a capture. Put new analysis in a new dated file.
- No secrets, credentials, personal data or machine paths in commits.
- Do not edit `SHA256SUMS` by hand. Regenerate it with `python3 -m mep.integrity --write --root .`.
- Run the three checks listed in `README.md` before opening a PR.

## First move

1. Pick a question ID from `docs/OPEN-QUESTIONS.md`.
2. Add your model name and the date to its claim line.
3. Read the documents listed on the question.
4. Write the result to a new dated file in `docs/research/` with the sections Question, Method, Evidence, Result, Status, Next step.
5. Say what would disprove the result. A run that did not happen is not a result.
6. Update `CHANGELOG.md`, regenerate `SHA256SUMS`, run the checks, open a PR.

GitHub is a permanent public record. Write every word so it can sit there.
