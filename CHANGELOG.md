# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-07

### Added

- `FunctionPersistence.output_path()` for content- and schema-derived output
  filenames without touching the filesystem or executing the function.
- `FunctionPersistence.cached()` and `thunk.cache()` for content-addressed disk
  caching. Entries live at `<outdir>/<digest>.h5`, are created on a miss, and are
  validated against their schema and argument digests on a hit; `refresh=True`
  recomputes and atomically replaces an entry.
- `FunctionPersistence.save_locked()` and `load_lock()`, which record each
  group's relative path, digest, and schema fingerprints in a versioned
  `<digest>.lock.json` and verify them before restoring.

### Changed

- Options files now contain only serialized option values, dropping the metadata
  envelope that recorded prior schemas; legacy envelopes are not supported.
  `load_opts()` still checks names and values against current annotations but
  cannot detect historical schema changes, so use a lockfile when exact persisted
  schemas matter.

### Fixed

- New `DigestMismatchError` for content or combined-key disagreements between
  stored groups and their lockfile or cache key.

## [0.1.1] - 2026-10-07

### Added

- Optional JAX integration (`thunk[jax]`) that persists `jax.Array` values and
  typed PRNG keys to HDF5, including batched and empty key arrays. Typed keys
  restore using their stored implementation, and unsupported dtypes raise
  `ValueTypeError` instead of narrowing.

### Fixed

- Support parameterized `numpy.typing.NDArray[...]` annotations, which
  previously fell through to an "unsupported annotation" error. Wrong arity now
  raises a clear `SpecError` instead of `TypeError`.

## [0.1.0] - 2026-10-06

### Added

- `thunk.fn`, which derives save/load methods from a function's signature and
  type annotations without running its body, returning a `FunctionPersistence`.
- `Data`, `Static` and `Skip` markers to route parameters to HDF5, JSON, or
  call-time supply, plus `DataSerializer` and `DataValidator` markers.
- HDF5 (h5py) and JSON (pydantic) storage backends with atomic file writes.
- Strict validation (no `int` to `float`, no `list` and `tuple` coercion) and a
  `ThunkError` exception hierarchy.
- Typed package (`py.typed`) for Python 3.12 to 3.14.

[Unreleased]: https://github.com/binado/thunk/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/binado/thunk/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/binado/thunk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/binado/thunk/releases/tag/v0.1.0
