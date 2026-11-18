# MDE

MDE is a research repo for Model Domain Expansion. It asks whether a model
can work in its own weights instead of only through text tokens. The line it
follows is W-MCP: a weight-space version of the Model Context Protocol, where
layers, residual directions and low-rank adapters become typed endpoints
instead of tokens in a prompt.

This repo holds captures, evidence and open questions for that line. It is for
research models and the people who run them. It does not ship a weight-space
protocol and it does not change MEP v0.1.

## Where this came from

MDE began as a copy of [ethos](https://github.com/Moissanite-Labs/ethos) at
commit `afdb3a4`. The ethos repo captured the first W-MCP notes on 2026-11-14
and then left the weight-space track out of its roadmap. MDE carries that
track forward. The research files are kept verbatim, with processed analysis
next to every raw transcript.

## Start here

- [START-HERE.md](START-HERE.md): the entry point for a research model.
- [docs/ORIENTATION.md](docs/ORIENTATION.md): mission, settled facts, hard vetoes, working rules.
- [docs/research/INDEX.md](docs/research/INDEX.md): every captured file and its standing.
- [docs/OPEN-QUESTIONS.md](docs/OPEN-QUESTIONS.md): numbered questions ready to pick up.

## What is settled and what is not

Settled within the captured evidence:

- AlphaEdit's null-space projection works for declarative knowledge edits. It is a published result with a defined scope.
- The compositor prototype's corrected invariant holds at about 3.7e-15 on a toy model. Its own printed check measures the wrong product and reports 6.93.
- The SC-WSP brief overstates its claims. Its headline product is written with the wrong orientation. It is preserved, not adopted.
- A portable weight wire that works across model families does not exist. The gauge problem is open.
- No shipped inference runtime projects adapters at load time. Without that, a weight-space certificate proves nothing about what the host will do.

Open: how to bind a delta to a checkpoint family, how to verify an untrusted
delta before it reaches a GPU, how to keep concurrent adapters from
overwriting each other, and whether the reported benchmark gains reproduce.

## Repo layout

```
START-HERE.md       entry point for research models
AGENTS.md           rules a coding or research harness reads on arrival
docs/ORIENTATION.md mission, evidence bar, working rules
docs/OPEN-QUESTIONS.md   numbered research tasks
docs/research/       captures, processed notes, prototypes
docs/proposal/       the ethos convergence proposal
MEP.md, v0.1/, mep/, template/, tests/    the ethos serving substrate
```

The substrate stays because MDE is still an MEP server. It serves this
research corpus the same way an ethos serves a person. The weight-space
designs in `docs/` are research notes, not MEP behavior.

## Verify this repo

```
python3 -m unittest discover -s tests -v
python3 v0.1/test_fixtures.py
python3 -m mep.integrity --root . --verify
```

The substrate needs no third-party Python packages. The prototype scripts
need numpy and are not part of the test suite.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Plain language, dated entries,
evidence over claims, no personal data.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
