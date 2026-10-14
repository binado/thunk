"""Plain finite JSON storage. Reconstruction is shared with HDF5."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._errors import StorageFormatError


def write_opts(path: Path, encoded: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(encoded), f, indent=2, allow_nan=False)
        f.write("\n")


def read_opts(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_invalid_constant
        )
        json.dumps(doc, allow_nan=False)
    except (ValueError, UnicodeDecodeError) as exc:
        raise StorageFormatError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise StorageFormatError(f"{path}: options must be a JSON object")
    return doc


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant {value}")
