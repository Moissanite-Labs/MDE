# Agent instructions

Read `START-HERE.md` first, then `docs/ORIENTATION.md`. The research index and
the open questions come next. Do not start a weight-space experiment from a
raw transcript alone.

- Treat every W-MCP design in this repo as research. None of it is shipped MEP behavior.
- Label claims `verified`, `source claim` or `proposed`, and attach the evidence.
- Keep raw captures verbatim. Put new analysis in a new dated file.
- Claim an open question in `docs/OPEN-QUESTIONS.md` before working on it.
- Write results as Question, Method, Evidence, Result, Status, Next step.
- State what would disprove the result. Record a failed experiment as a result.
- No secrets, credentials, personal data or machine paths in commits or docs.
- Do not edit `SHA256SUMS` by hand. Regenerate it with `python3 -m mep.integrity --write --root .`.
- Run the three checks from `README.md` before opening a PR.

GitHub is a permanent public record. Write every word so it can sit there.
