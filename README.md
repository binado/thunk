# thunk

Generate automatic save and load methods for functions that manipulate arrays

## Usage

`thunk.fn` derives save/load methods from a callable's type annotations. It
inspects the signature and never runs the function body.

```python
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import thunk


@dataclass(frozen=True)
class Population:
    positions: np.ndarray
    labels: list[str]


def spectra(
    seeds: np.ndarray,
    /,
    population: Population,
    *,
    bins: int = 128,
    chunk_size: Annotated[int, thunk.Skip()] = 4096,
) -> np.ndarray: ...


pfn = thunk.fn(spectra)

# Bind arguments and split them into persisted groups (does not run spectra).
inputs, opts = pfn.flatten(seeds, population, bins=256)

pfn.save_inputs("inputs.h5", inputs)
pfn.save_opts("opts.json", opts)
# ... or flatten and save both in one step:
pfn.save("inputs.h5", "opts.json", seeds, population, bins=256)

# Restore, optionally edit the configuration, and run.
inputs = pfn.load_inputs("inputs.h5")
opts = pfn.load_opts("opts.json")
result = pfn(inputs, opts)

# Or restore a legal Python call without executing spectra.
args, kwargs = pfn.load(inputs="inputs.h5", opts="opts.json")
result = spectra(*args, **kwargs)

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
`Literal[...]`, `T | None`, `list[T]`, `tuple[...]`, `dict[str, T]`, and
dataclasses of those. Values are validated strictly (no `int` → `float`, no
`list` ↔ `tuple`). Types outside that set can use
`Data(DataSerializer(...), DataValidator(...))` or
`Static(PlainSerializer(...), PlainValidator(...))`.

Loading takes `extras="forbid" | "ignore"` for stored names that are no longer
parameters and `missing="raise" | "default"` for parameters absent from a file
(defaults are filled with a warning).

## Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.



## Development

```console
just fmt        # uv run ruff check --fix . && uv run ruff format .
just lint       # uv run ruff check .
just typecheck  # uv run ty check
just test       # uv run pytest
just hooks      # uv run prek run --all-files
just check      # full validation, tests, and package build
```

## Update from the template

The template runs `uv lock`, so updates must explicitly trust it:

```console
uvx copier@9.18.2 update --trust
just check
```


## Publishing

Configure `pypi` and `testpypi` GitHub environments with trusted publishers.
Push a tag matching the version in `pyproject.toml`, such as `v0.1.0`, to
publish to PyPI. Run the Release workflow manually to publish to TestPyPI.
