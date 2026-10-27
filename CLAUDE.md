# ArcCharCalc

Python tool that extracts breakdown characteristics of MV insulators from
oscilloscope measurements (LeCroy CSV). It replaces Petros's Excel workflow and
must reproduce its numbers. Thesis-wide rules: see `../CLAUDE.md`.

## How I want to work
- I'm learning Python, but I want to move faster: you may write the code.
- Work in small steps: one function or one change at a time.
  Before writing, show a short plan and wait for my OK.
- After writing, explain briefly what the code does. Explain line by line
  anything non-obvious (new syntax, numpy tricks, Python idioms).
- Prefer simple, readable code over clever code. Ask before adding a new dependency.
- Every function gets a docstring and type hints.
- Every numerical result must be checked against Petros's Excel values
  (`../data/reference/`) before it counts as done. Propose how to check it.
- When there is an error, first show me how to read the traceback, then fix it.
- Answer in Greek. Code, names and comments in English.

## Project layout
- `src/arccharcalc/__init__.py`: `main()`, the entry point (`[project.scripts]`).
- Other modules in `src/arccharcalc/`: ONLY definitions (functions, classes,
  constants). No code at module top level.
- Data are OUTSIDE this repo, in `../data/raw/` (read-only). Never copy data into
  this repo, and never hard-code data paths in modules.
- Outputs go to `../results/`, never inside this repo.

## Environment (uv)
- Run: `uv run arccharcalc`
- Add a package: `uv add <name>`. Never use `pip install`.
- Recreate the environment: `uv sync`
- Never edit `uv.lock` by hand.

## Git
- Never run `git commit` or `git push`. I do that myself. You may suggest a
  commit message (English, imperative, e.g. "Add CSV reader").
- `git status`, `git diff` and `git log` are fine.

## Verification
- Every number we compute must be checked against Petros's Excel values
  (`../data/reference/`) before it counts as done.