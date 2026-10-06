"""JSON encoding of ``Static`` parameters through pydantic ``TypeAdapter``."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, TypeAdapter, ValidationError

from ._errors import StorageFormatError
from ._signature import Param
from ._spec import (
    DataclassNode,
    DictNode,
    FixedTupleNode,
    ListNode,
    Node,
    OptionalNode,
    Scalar,
    VarTupleNode,
)

STORAGE_VERSION = 1
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_CONFIG = ConfigDict(ser_json_inf_nan="strings")


def _revive_floats(node: Node, raw: Any) -> Any:
    """Turn ``"NaN"``/``"Infinity"`` strings back into floats at float positions.

    Pydantic's strict JSON mode does not accept those strings for ``float``, so
    the document is rewritten with real floats (emitted as bare ``NaN`` tokens,
    which pydantic's JSON parser accepts) before strict validation.
    """
    match node:
        case Scalar(type=tp) if tp is float:
            return _NONFINITE.get(raw, raw) if isinstance(raw, str) else raw
        case OptionalNode(inner=inner):
            return None if raw is None else _revive_floats(inner, raw)
        case ListNode(inner=inner) | VarTupleNode(inner=inner):
            if isinstance(raw, list):
                return [_revive_floats(inner, x) for x in raw]
        case FixedTupleNode(items=items):
            if isinstance(raw, list) and len(raw) == len(items):
                return [_revive_floats(n, x) for n, x in zip(items, raw, strict=True)]
        case DictNode(value=inner):
            if isinstance(raw, dict):
                return {k: _revive_floats(inner, v) for k, v in raw.items()}
        case DataclassNode(fields=fields):
            if isinstance(raw, dict):
                by_name = dict(fields)
                return {
                    k: _revive_floats(by_name[k], v) if k in by_name else v
                    for k, v in raw.items()
                }
    return raw


class OptsCodec:
    """Encode and decode the opts group of one callable."""

    def __init__(self, params: tuple[Param, ...]) -> None:
        self.params = params
        self._adapters: dict[str, TypeAdapter[Any]] = {}

    def adapter(self, param: Param) -> TypeAdapter[Any]:
        adapter = self._adapters.get(param.name)
        if adapter is None:
            # Wrapped in a 1-tuple: pydantic refuses ``config=`` for a bare
            # dataclass, but still applies it to dataclasses nested in a wrapper.
            adapter = TypeAdapter(tuple[param.pydantic_annotation], config=_CONFIG)
            self._adapters[param.name] = adapter
        return adapter

    def encode(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and encode ``values`` (in signature order) to JSON values."""
        out: dict[str, Any] = {}
        for p in self.params:
            assert p.node is not None
            p.node.validate(values[p.name], p.name)
            out[p.name] = json.loads(self.adapter(p).dump_json((values[p.name],)))[0]
        return out

    def decode_one(self, param: Param, raw: Any, where: str) -> Any:
        assert param.node is not None
        revived = _revive_floats(param.node, raw)
        try:
            (value,) = self.adapter(param).validate_json(
                json.dumps([revived], allow_nan=True), strict=True
            )
            param.node.validate(value, param.name)
        except (ValidationError, TypeError) as exc:
            raise StorageFormatError(
                f"{where}: stored value of {param.name!r} is invalid: {exc}"
            ) from exc
        return value


def write_opts(
    path: Path, fingerprints: Mapping[str, str], encoded: Mapping[str, Any]
) -> None:
    envelope = {
        "thunk_format": "opts",
        "storage_version": STORAGE_VERSION,
        "fingerprints": dict(fingerprints),
        "values": dict(encoded),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, allow_nan=False)
        f.write("\n")


def read_opts_envelope(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StorageFormatError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("thunk_format") != "opts":
        raise StorageFormatError(f"{path}: not a thunk 'opts' file")
    if doc.get("storage_version") != STORAGE_VERSION:
        raise StorageFormatError(
            f"{path}: unsupported storage_version {doc.get('storage_version')!r}"
        )
    fps, values = doc.get("fingerprints"), doc.get("values")
    if not isinstance(fps, dict) or not isinstance(values, dict):
        raise StorageFormatError(f"{path}: malformed envelope")
    return fps, values
