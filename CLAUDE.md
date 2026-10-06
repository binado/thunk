# thunk

Generates save/load methods for functions that manipulate arrays. `thunk.fn(f)`
inspects `f`'s signature and type annotations (never runs the body) and returns
a `FunctionPersistence` that splits arguments into HDF5 (`Data`) and JSON
(`Static`) groups; `Skip` parameters are supplied at call time. See `README.md`
for the user-facing workflow.

## Commands

Tooling is uv + just. Python >= 3.12.

- `just setup` — `uv sync --all-groups`
- `just test [args]` — pytest (e.g. `just test tests/test_spec.py -k name`)
- `just lint` / `just fmt` / `just fmt-check` — ruff
- `just typecheck` — `ty check` (covers `src` and `tests`)
- `just hooks` — run all prek hooks (`prek.toml`)
- `just check` — lint, format check, types, tests, build, lockfile check; run
  before committing. CI (`.github/workflows`) runs the same checks.

## Layout

Package is `src/thunk/`; private modules are underscore-prefixed and the public
API is re-exported in `__init__.py` (`__all__`).

- `_persistence.py` — `fn` and `FunctionPersistence` (flatten/save/load API)
- `_signature.py` — signature + annotation resolution into roles
- `_spec.py` — per-parameter specs derived from annotations
- `_markers.py` — `Data`, `Static`, `Skip`, serializer/validator markers
- `_hdf5.py` / `_opts_json.py` — storage backends (h5py / pydantic JSON)
- `_atomic.py` — atomic file writes; `_fingerprint.py`; `_errors.py` — `ThunkError` hierarchy

## Conventions

- Validation is strict: no `int` → `float`, no `list` ↔ `tuple` coercion.
- New errors subclass `ThunkError` in `_errors.py` and are exported.
- Tests live in `tests/`; shared helpers in `tests/helper_funcs.py`. Add a
  regression test with every fix.
- Ruff rules: `E4 E7 E9 F I UP`, target py312. Keep `ty check` clean.
- Keep `uv.lock` in sync with `pyproject.toml` (`uv lock`).

## Git

- Conventional Commits (`feat(scope): ...`, `fix: ...`).
- Never commit to `main`; work on a branch, commit and push when a task is done.
- Stage files by name, not `git add -A`.
