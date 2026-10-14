# thunk

Generate automatic save and load methods for functions that manipulate arrays

## Installation

```console
uv add thunk
# or
pip install thunk
```

## Usage

`thunk.fn` inspects a callable's signature without running its body or checking
default values. Actual argument values determine storage on each operation;
annotations are optional hints for reconstruction when loading.

```python
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import thunk


@dataclass(frozen=True)
class Params:
    y: np.ndarray
    labels: list[str]


def simulator(
    x: np.ndarray,
    /,
    params: Params,
    *,
    seed: int = 0,
    chunk_size: Annotated[int, thunk.Skip()] = 4096,
) -> np.ndarray: ...


pfn = thunk.fn(simulator)

# Bind arguments and split them into persisted groups (does not run simulator).
inputs, opts = pfn.flatten(x, params, seed=42)

pfn.save_inputs("inputs.h5", inputs)
pfn.save_opts("opts.json", opts)
# ... or flatten and save both in one step:
pfn.save("inputs.h5", "opts.json", x, params, seed=42)

# Restore, optionally edit the configuration, and run.
inputs = pfn.load_inputs("inputs.h5")
opts = pfn.load_opts("opts.json")
result = pfn(inputs, opts)

# Or restore a legal Python call without executing simulator.
args, kwargs = pfn.load(inputs="inputs.h5", opts="opts.json")
result = simulator(*args, **kwargs)

pfn.save_output("output.h5", result, inputs=inputs, opts=opts)
restored = pfn.load_output("output.h5")
```

Roles decide where each parameter is stored:

| Role | Group | Storage |
| --- | --- | --- |
| `Data()` (inferred for values containing arrays or nonfinite floats) | `inputs` | HDF5 |
| `Static()` (inferred for other supported values) | `opts` | Plain finite JSON |
| `Skip()` | neither | supplied at call time |

Supported runtime values are `bool`, `int`, `float`, `str`, `None`, NumPy/JAX
arrays, lists, tuples, string-keyed dictionaries, and dataclasses containing
these values. Containers may be empty or heterogeneous. Cycles, unsupported
objects or dictionary keys, and dataclass `InitVar` / `init=False` fields are
rejected with a nested path. Ordinary annotations (including `Any`, unresolved
forward references, unions, and scalar constraints) do not validate values.
Variadic parameters and nested or conflicting persistence markers are rejected.

Explicit roles override inference. `Static()` requires finite JSON values and
rejects arrays and nonfinite floats. Custom codecs can convert otherwise
unsupported values: use `Data(DataSerializer(...), DataDeserializer(...))` or
`Static(PlainSerializer(...), PlainValidator(...))`. `DataValidator` remains an
alias for `DataDeserializer`, and the `validator=` argument remains supported.
Data serializers return nested dictionaries of scalars and NumPy arrays; Static
serializers return finite JSON-compatible values. Codecs receive actual values
without annotation checks; deserializer results are accepted as returned.

JSON tuples become lists and dataclasses become field dictionaries. Compatible
structural hints can restore tuples and dataclasses recursively, including
constructor defaults. Optional hints apply to non-`None` values; other unions
are ignored. Incompatible shapes and unsupported hints leave native decoded
values unchanged. Scalar values are never coerced. A selected dataclass
constructor that fails raises `ReconstructionError`. Custom deserializers take
precedence over hints. HDF5 tags preserve list/tuple and NumPy/JAX array identity;
dataclasses still require a current hint. No class named in a file is imported.
Round trips therefore need not preserve the original Python structure without
appropriate hints or custom codecs.

```python
def identity(x):
    return x


p = thunk.fn(identity)
p.flatten(1)  # ({}, {"x": 1})
p.flatten(np.arange(3))  # ({"x": array(...)}, {})
p.flatten({"n": float("inf")})  # Data, including nested nonfinite values
```

`flatten()` applies defaults, drops Skip values, preserves array identity, and
performs no custom serialization or device transfers. Inference is local to
each operation: alternating calls cannot change how earlier groups are saved.
Individual group savers use supplied membership for unmarked parameters and
check explicit roles, names, and backend compatibility. `__call__()` assembles
and executes supplied values; it does not reconstruct them.

Loading takes `extras="forbid" | "ignore"` for stored names that are no longer
parameters. Explicit role changes always raise. Individual group loaders check
absence only for parameters explicitly assigned to that group. `load()` checks
completeness across both files, rejects duplicate names, and uses
`missing="raise" | "default"` to either reject absent persisted parameters or
insert current defaults with a warning. Required parameters without defaults
still raise. `load_lock()` always requires the exact persisted names.

## Deterministic output paths and locked inputs

Options files remain plain mappings (for example, `{"seed": 42}`), without a
metadata envelope. Existing plain JSON can be loaded under the new hint rules.
HDF5 storage, locks, and content digests now use version 2. Version-1 HDF5 and
lockfiles are rejected; regenerate them using current thunk. There is no legacy
reader or migration tool. Use a lockfile to verify concrete representations and
encoded content before any custom deserializer or dataclass constructor runs:

```python
lock_path = pfn.save_locked("inputs.h5", "opts.json", x, params, seed=42)
inputs, opts = pfn.load_lock(lock_path)

output_path = pfn.output_path(inputs, opts, base_dir="outputs/simulator-v1")
# Create the output directory before saving; thunk does not create directories.
result = pfn(inputs, opts)
pfn.save_output(output_path, result, inputs=inputs, opts=opts)
```

`output_path()` validates both groups and returns `<full SHA-256 digest>.h5`,
optionally under `base_dir`. It never accesses files or executes the function,
and does not require a supported return annotation. The key includes persisted
concrete roles, runtime structures, codec identities, and values, excluding
all ordinary annotations and reconstruction hints, function identity,
`Skip` arguments, and file paths. Use directories such as `simulator-v1` to
separate computations and revisions. This method does not execute the function or reuse cached results; use
`cached()` or `thunk.cache()` for that. Hashes are recomputed so mutations are
reflected.

Parameter groups are hashed in signature order; nested dictionary order remains
significant. Array layout and byte order are normalized, while dtype, shape, and
values remain significant. Custom serializers must produce deterministic
representations. When both groups are passed to `save_output()`, its HDF5
attributes include `digest`, `digest_version`, and the two group digests.

`save_locked()` returns `<digest>.lock.json` beside the options file. Its strict,
versioned metadata records each group's relative path, digest, concrete
representations, codec identities, and fingerprints. Paths may contain `..`; relocating the files together preserves
the lock. Input HDF5 files also retain their fingerprints. `load_lock()` verifies
schemas and content before returning `(inputs, opts)`; it offers no extras or
default-filling policies. Content or key mismatches raise `DigestMismatchError`,
schema mismatches raise `SchemaMismatchError`, malformed metadata raises
`StorageFormatError`, and missing files raise `FileNotFoundError`.

All three files are prepared before replacing any destination, and the lockfile
is published last. Each replacement is atomic; the group is not a filesystem
transaction. Parent directories must already exist, destinations must be
distinct, and older lockfiles remain in place. Editing either referenced file
requires a new lockfile. Both `save()` and `save_locked()` forward all function
keyword arguments, including one named `lockfile`.

## Disk caching

Cache a function's output using its persisted arguments:

```python
cached_simulator = thunk.cache(simulator, outdir=".cache/simulator-v1")
result = cached_simulator(x, params, seed=42, chunk_size=4096)
# The same persisted arguments load the saved result without running simulator.
result = cached_simulator(x, params, seed=42, chunk_size=4096)
```

Or use already flattened or restored groups:

```python
inputs, opts = pfn.flatten(x, params, seed=42)
result = pfn.cached(
    inputs,
    opts,
    outdir=".cache/simulator-v1",
    skipped={"chunk_size": 4096},
)
```

Both interfaces share entries at `<outdir>/<digest>.h5` and require an explicit
`outdir` (a string or path-like object). Relative and absolute paths are supported.
Directories are created automatically on a miss.

Each output directory must identify one computation and revision, including bound
instance state, partial arguments omitted from the exposed signature, captured
values, and external dependencies that affect results. Use a new directory when
those change. There is no automatic function, source, or package-version hashing.
Persisted arguments must determine results together with that computation and
revision; `Skip` values must only control execution details that do not affect
results. Persist random
seeds or keys explicitly. Do not mutate inputs during execution, and do not rely
on side effects being replayed on cache hits.

Return annotations are optional: output structure is inferred from actual values
and stored in HDF5. Explicit Data output codecs use the same nested dictionary
contract; Static output codecs store their finite JSON payload inside HDF5.
Return `Skip` is rejected. Hits check stored structure, codec compatibility,
output content integrity, and recorded argument digests before reconstruction.
Changing reconstruction hints can change a restored cache hit without changing
its key. Refresh caches or choose a new directory when hint changes require
different reconstruction behavior. Invalid files, schema or
digest mismatches, and permission errors propagate rather than triggering
recomputation. Only absent entries are misses.

Pass `refresh=True` to recompute and atomically replace an entry. Failed
computation or saving preserves an existing entry. For `thunk.cache`, cache
controls are fixed when constructing the wrapper: a wrapper created with
`refresh=True` recomputes on every invocation. All arguments passed to the
wrapper itself belong to the underlying function, even arguments named
`outdir` or `refresh`. Refresh only replaces invoked entries;
a new output directory starts a separate cache without deleting existing entries.

Caching saves outputs only; input files and lockfiles are optional. Concurrent
misses may execute the function more than once, with the last successful atomic
replacement winning. There is no eviction, expiration, or concurrency locking.

## JAX arrays and random keys

Install the optional integration (JAX 0.11.2 or newer):

```console
uv add 'thunk[jax]'
# or
pip install 'thunk[jax]'
```

NumPy arrays, JAX arrays, and typed PRNG keys infer `Data` from their runtime
values, including inside containers and dataclasses. Annotations are optional;
array identity is determined by stored metadata.

```python
import jax
import jax.numpy as jnp
import thunk


def simulate(x: jax.Array, key: jax.Array, *, scale: float = 0.1) -> jax.Array:
    return x + scale * jax.random.normal(key, x.shape, dtype=x.dtype)


pfn = thunk.fn(simulate)  # jax.jit(simulate) also works
x = jnp.zeros((100, 3), dtype=jnp.float32)
key = jax.random.key(42, impl="threefry2x32")
pfn.save("inputs.h5", "opts.json", x, key, scale=0.2)
inputs = pfn.load_inputs("inputs.h5")
opts = pfn.load_opts("opts.json")
result = pfn(inputs, opts)
pfn.save_output("output.h5", result, inputs=inputs, opts=opts)
```

Values, shapes, numeric dtypes, and typed-key implementations are preserved.
Supported numeric dtypes are bool, int8/16/32/64, uint8/16/32/64,
float16/32/64, and complex64/128. Extended dtypes such as bfloat16 and float8
are rejected. Loading int64, uint64, float64, or complex128 requires
`JAX_ENABLE_X64=1` or `jax.config.update("jax_enable_x64", True)`; thunk raises
`ValueTypeError` rather than narrowing the stored dtype.

Typed keys support `threefry2x32`, `threefry4x32`, `philox2x32`, `philox4x32`,
`rbg`, and `unsafe_rbg`, including batched and empty key arrays. Restoration
uses the stored implementation regardless of JAX's current default RNG.
Legacy `uint32` keys are stored as ordinary numeric arrays. Custom RNG
implementations are unsupported.

Weakly typed arrays (for example `jnp.asarray(1)`) are rejected: supply an
explicit dtype such as `jnp.asarray(1, dtype=jnp.int32)`. Tracers, deleted
arrays, and arrays not fully addressable by the current process are also
rejected with a parameter or nested-field path. Persist outside JAX
transformations, before deletion or buffer donation, and gather distributed
data on the saving process first.

`flatten()` retains array identity and performs no transfers. Saving and
input content hashing synchronize and transfer array data to the host;
loading places arrays on JAX's default device. Device placement and sharding
are not restored. Custom pytrees and distributed checkpointing are outside
this integration's scope. Importing thunk or compiling NumPy annotations
does not import JAX. Version-1 files must be regenerated as described above.

## Contributing

### Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.

### Development

```console
just fmt        # uv run ruff check --fix . && uv run ruff format .
just lint       # uv run ruff check .
just typecheck  # uv run ty check
just test       # uv run pytest
just hooks      # uv run prek run --all-files
just check      # full validation, tests, and package build
```

### Update from the template

The template runs `uv lock`, so updates must explicitly trust it:

```console
uvx copier@9.18.2 update --trust
just check
```

### Publishing

Configure `pypi` and `testpypi` GitHub environments with trusted publishers.
Push a tag matching the version in `pyproject.toml`, such as `v0.1.0`, to
publish to PyPI. Run the Release workflow manually to publish to TestPyPI.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
