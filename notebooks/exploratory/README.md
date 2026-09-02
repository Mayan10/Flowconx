# Exploratory notebooks

**Non-authoritative.** Nothing in this directory produces a number that appears
in the paper, and nothing here is run by CI, by `make repro-small`, or by
`make repro-full`.

Every result in the paper comes from `python -m flowconx.run --config <yaml>`
and is written to `results/` with its seed, config hash, git commit and
environment recorded. If something here disagrees with `results/`, `results/`
is right.

Notebooks are kept because the exploration was real and a reviewer may want to
see how a decision was reached. They are not kept as evidence.
