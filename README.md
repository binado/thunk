# thunk

Generate automatic save and load methods for functions that manipulate arrays

## Installation

```console
uv add thunk
# or
pip install thunk
```

## Usage

`thunk.fn` derives save/load methods from a callable's type annotations. It
inspects the signature and never runs the function body.

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
| `Data()` (default when the annotation contains arrays) | `inputs` | HDF5 |
| `Static()` (default otherwise) | `opts` | JSON, via pydantic |
| `Skip()` | neither | supplied at call time |

Supported annotations: `bool`, `int`, `float`, `str`, `None`, `np.ndarray`,
`jax.Array` (with the JAX extra), `Literal[...]`, `T | None`, `list[T]`, `tuple[...]`, `dict[str, T]`, and
dataclasses of those. Values are validated strictly (no `int` → `float`, no
`list` ↔ `tuple`). Types outside that set can use
`Data(DataSerializer(...), DataValidator(...))` or
`Static(PlainSerializer(...), PlainValidator(...))`.

Pydantic `BaseModel` subclasses (including nested models and models in the
supported containers) default to `Static` and round-trip through JSON as model
instances. Their fields use Pydantic's validation, serializers, and model config;
loading uses strict validation. Pass model instances when saving. To store models
as `Data`, supply a custom serializer/validator pair.

Loading takes `extras="forbid" | "ignore"` for stored names that are no longer
parameters and `missing="raise" | "default"` for parameters absent from a file
(defaults are filled with a warning).

## JAX arrays and random keys

Install the optional integration (JAX 0.11.2 or newer):

```console
uv add 'thunk[jax]'
# or
pip install 'thunk[jax]'
```

Annotate arrays and typed PRNG keys with `jax.Array`. Both infer `Data`,
including inside the supported containers and dataclasses. NumPy and JAX
annotations require their respective array types; `jax.typing.ArrayLike` and
mixed-array unions are unsupported.

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
does not import JAX, and existing NumPy files remain compatible.

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
