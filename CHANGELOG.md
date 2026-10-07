# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/binado/thunk/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/binado/thunk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/binado/thunk/releases/tag/v0.1.0
