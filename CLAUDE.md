# ArcCharCalc

Python tool that extracts breakdown characteristics of MV insulators from
oscilloscope measurements (LeCroy CSV). It replaces Petros's Excel workflow and
must reproduce its numbers. Thesis-wide rules: see `../CLAUDE.md`.

## How I want to work (I'm learning Python)
- This is my first real Python project. My goal is to learn, not just to finish.
- Default: explain, give hints, small examples and reviews. Do NOT write complete
  functions or modules unless I explicitly say "write it".
- Before changing any file, tell me what you will change and why, and wait.
- When I get an error, help me read the traceback first. Don't jump to the fix.
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