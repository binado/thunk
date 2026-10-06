"""Node-driven HDF5 writer and reader."""

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ._errors import SerializerContractError, StorageFormatError, ValueTypeError
from ._spec import (
    Array,
    CustomData,
    CustomStatic,
    DataclassNode,
    DictNode,
    FixedTupleNode,
    ListNode,
    LiteralNode,
    Node,
    NoneNode,
    OptionalNode,
    Scalar,
    VarTupleNode,
    qualified_name,
)

STORAGE_VERSION = 1
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}

# -- file envelope -------------------------------------------------------


def write_envelope(
    f: h5py.File, kind: str, fingerprints: Mapping[str, str], **extra: str
) -> None:
    f.attrs["thunk_format"] = kind
    f.attrs["storage_version"] = STORAGE_VERSION
    f.attrs["fingerprints"] = json.dumps(dict(fingerprints), sort_keys=True)
    for key, value in extra.items():
        f.attrs[key] = value


def open_checked(path: Path, kind: str) -> tuple[h5py.File, dict[str, str]]:
    """Open ``path`` read-only and validate its envelope; caller closes it."""
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise StorageFormatError(f"{path}: not a readable HDF5 file: {exc}") from exc
    try:
        found = _attr(f, "thunk_format", path)
        if found != kind:
            raise StorageFormatError(
                f"{path}: expected a thunk {kind!r} file, found {found!r}"
            )
        version = _attr(f, "storage_version", path)
        if version != STORAGE_VERSION:
            raise StorageFormatError(
                f"{path}: unsupported storage_version {version!r} "
                f"(this thunk reads {STORAGE_VERSION})"
            )
        try:
            fps = json.loads(_attr(f, "fingerprints", path))
        except json.JSONDecodeError as exc:
            raise StorageFormatError(f"{path}: malformed fingerprints") from exc
        if not isinstance(fps, dict):
            raise StorageFormatError(f"{path}: malformed fingerprints")
        return f, fps
    except BaseException:
        f.close()
        raise


def _attr(obj: Any, name: str, where: Any) -> Any:
    try:
        value = obj.attrs[name]
    except KeyError:
        raise StorageFormatError(f"{where}: missing attribute {name!r}") from None
    return value.decode() if isinstance(value, bytes) else value


# -- key escaping --------------------------------------------------------


def escape_key(key: str) -> str:
    out = ["k"]
    for ch in key:
        if ch in "/%." or ord(ch) < 32 or ch == "\x7f":
            out.append(f"%{ord(ch):02X}")
        else:
            out.append(ch)
    return "".join(out)


_ESCAPE = re.compile(r"%([0-9A-F]{2})")


def unescape_key(name: str, where: str) -> str:
    if not name.startswith("k"):
        raise StorageFormatError(f"{where}: malformed dict key {name!r}")
    body = name[1:]
    if re.search(r"%(?![0-9A-F]{2})", body):
        raise StorageFormatError(f"{where}: malformed dict key {name!r}")
    return _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), body)


# -- scalars -------------------------------------------------------------


def _encode_scalar(value: Any) -> str:
    if type(value) is float and not math.isfinite(value):
        value = (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return json.dumps(value, allow_nan=False)


def _write_scalar(parent: h5py.Group, name: str, value: Any) -> None:
    g = parent.create_group(name)
    g.attrs["kind"] = "scalar"
    g.attrs["type"] = type(value).__name__
    g.attrs["value"] = _encode_scalar(value)


def _read_scalar(g: h5py.Group, tp: type | None, where: str) -> Any:
    if tp is None:
        name = _attr(g, "type", where)
        tp = {"bool": bool, "int": int, "float": float, "str": str}.get(name)
        if tp is None:
            raise StorageFormatError(f"{where}: unknown scalar type {name!r}")
    try:
        value = json.loads(_attr(g, "value", where))
    except json.JSONDecodeError as exc:
        raise StorageFormatError(f"{where}: malformed scalar value") from exc
    if tp is float and isinstance(value, str) and value in _NONFINITE:
        return _NONFINITE[value]
    if type(value) is not tp:
        raise StorageFormatError(
            f"{where}: stored {type(value).__name__} where {tp.__name__} is expected"
        )
    return value


# -- arrays --------------------------------------------------------------


def _write_array(parent: h5py.Group, name: str, value: np.ndarray) -> None:
    if value.dtype.kind == "U":
        arr = np.asarray(value, order="C")
        nchar = max(arr.dtype.itemsize // 4, 1)
        codes = (
            np.asarray(arr.astype(f"<U{nchar}", copy=False), order="C")
            .reshape(-1)
            .view("<u4")
            .reshape((*arr.shape, nchar))
        )
        ds = parent.create_dataset(name, data=codes)
        ds.attrs["numpy_dtype"] = arr.dtype.str
    else:
        parent.create_dataset(name, data=value)


def _read_array(ds: h5py.Dataset, where: str) -> np.ndarray:
    if ds.dtype.kind == "O":
        raise StorageFormatError(
            f"{where}: object/variable-length datasets are unsupported"
        )
    data = np.asarray(ds[()])
    if "numpy_dtype" in ds.attrs:
        dtype = np.dtype(_attr(ds, "numpy_dtype", where))
        if dtype.kind != "U" or data.dtype != np.dtype("<u4") or data.ndim < 1:
            raise StorageFormatError(f"{where}: malformed unicode array")
        shape = data.shape[:-1]
        nchar = data.shape[-1]
        native = np.asarray(data, order="C").reshape(-1).view(f"<U{nchar}")
        return native.astype(dtype).reshape(shape)
    if data.dtype.kind not in "biufcS":
        raise StorageFormatError(f"{where}: unsupported dtype {data.dtype}")
    return data


# -- writer --------------------------------------------------------------


def write_node(parent: h5py.Group, name: str, node: Node, value: Any) -> None:
    """Write an already-validated ``value`` at ``parent[name]``."""
    match node:
        case Scalar() | LiteralNode():
            _write_scalar(parent, name, value)
        case NoneNode():
            _write_none(parent, name)
        case Array():
            _write_array(parent, name, value)
        case OptionalNode(inner=inner):
            if value is None:
                _write_none(parent, name)
            else:
                write_node(parent, name, inner, value)
        case ListNode(inner=inner) | VarTupleNode(inner=inner):
            kind = "list" if isinstance(node, ListNode) else "tuple"
            g = _group(parent, name, kind)
            for i, item in enumerate(value):
                write_node(g, str(i), inner, item)
        case FixedTupleNode(items=items):
            g = _group(parent, name, "tuple")
            for i, (sub, item) in enumerate(zip(items, value, strict=True)):
                write_node(g, str(i), sub, item)
        case DictNode(value=inner):
            g = _group(parent, name, "dict", track_order=True)
            for key, item in value.items():
                write_node(g, escape_key(key), inner, item)
        case DataclassNode(cls=cls, fields=fields):
            g = _group(parent, name, "dataclass")
            g.attrs["type"] = qualified_name(cls)
            for fname, sub in fields:
                write_node(g, fname, sub, getattr(value, fname))
        case CustomData(serializer=serializer):
            produced = serializer.func(value)
            g = _group(parent, name, "custom", track_order=True)
            _write_nested(g, produced, qualified_name(serializer.func))
        case CustomStatic():
            raise TypeError("custom static parameters are stored in the opts file")
        case _:
            raise TypeError(f"cannot write node {node!r}")


def _group(
    parent: h5py.Group, name: str, kind: str, *, track_order: bool = False
) -> h5py.Group:
    g = parent.create_group(name, track_order=track_order)
    g.attrs["kind"] = kind
    return g


def _write_none(parent: h5py.Group, name: str) -> None:
    _group(parent, name, "none")


def _write_nested(group: h5py.Group, produced: Any, who: str) -> None:
    """Write a custom serializer's nested dict, enforcing its contract."""
    if not isinstance(produced, Mapping):
        raise SerializerContractError(
            f"{who}: serializer must return a dict, got {type(produced).__name__}"
        )
    for key, item in produced.items():
        if (
            not isinstance(key, str)
            or not key
            or "/" in key
            or key in (".", "..")
            or "\0" in key
        ):
            raise SerializerContractError(f"{who}: invalid key {key!r}")
        if isinstance(item, Mapping):
            sub = _group(group, key, "dict", track_order=True)
            _write_nested(sub, item, who)
        elif isinstance(item, np.ndarray):
            if item.dtype.kind not in "biufcUS":
                raise SerializerContractError(
                    f"{who}: key {key!r} has unsupported array dtype {item.dtype}"
                )
            _write_array(group, key, item)
        elif type(item) in (bool, int, float, str):
            _write_scalar(group, key, item)
        else:
            raise SerializerContractError(
                f"{who}: key {key!r} has unsupported value type {type(item).__name__}"
            )


# -- reader --------------------------------------------------------------


def _expect_group(obj: Any, kind: str, where: str) -> h5py.Group:
    if not isinstance(obj, h5py.Group):
        raise StorageFormatError(f"{where}: expected a group of kind {kind!r}")
    found = _attr(obj, "kind", where)
    if found != kind:
        raise StorageFormatError(f"{where}: expected kind {kind!r}, found {found!r}")
    return obj


def _indexed(g: h5py.Group, where: str) -> list[Any]:
    names = list(g.keys())
    for n in names:
        if not (n.isascii() and n.isdigit() and str(int(n)) == n):
            raise StorageFormatError(f"{where}: malformed sequence index {n!r}")
    order = sorted(names, key=int)
    if [int(n) for n in order] != list(range(len(order))):
        raise StorageFormatError(f"{where}: sequence indices have gaps or duplicates")
    return [g[n] for n in order]


def read_node(obj: Any, node: Node, where: str) -> Any:
    match node:
        case Scalar(type=tp):
            return _read_scalar(_expect_group(obj, "scalar", where), tp, where)
        case LiteralNode():
            value = _read_scalar(_expect_group(obj, "scalar", where), None, where)
            try:
                node.validate(value, where)
            except TypeError as exc:
                raise StorageFormatError(str(exc)) from exc
            return value
        case NoneNode():
            _expect_group(obj, "none", where)
            return None
        case Array():
            if not isinstance(obj, h5py.Dataset):
                raise StorageFormatError(f"{where}: expected an array dataset")
            return _read_array(obj, where)
        case OptionalNode(inner=inner):
            if isinstance(obj, h5py.Group) and _attr(obj, "kind", where) == "none":
                return None
            return read_node(obj, inner, where)
        case ListNode(inner=inner):
            g = _expect_group(obj, "list", where)
            return [
                read_node(c, inner, f"{where}[{i}]")
                for i, c in enumerate(_indexed(g, where))
            ]
        case VarTupleNode(inner=inner):
            g = _expect_group(obj, "tuple", where)
            return tuple(
                read_node(c, inner, f"{where}[{i}]")
                for i, c in enumerate(_indexed(g, where))
            )
        case FixedTupleNode(items=items):
            g = _expect_group(obj, "tuple", where)
            children = _indexed(g, where)
            if len(children) != len(items):
                raise StorageFormatError(
                    f"{where}: expected {len(items)} tuple items, found {len(children)}"
                )
            return tuple(
                read_node(c, sub, f"{where}[{i}]")
                for i, (c, sub) in enumerate(zip(children, items, strict=True))
            )
        case DictNode(value=inner):
            g = _expect_group(obj, "dict", where)
            _require_ordered(g, where)
            result: dict[str, Any] = {}
            for name in g:
                key = unescape_key(name, where)
                result[key] = read_node(g[name], inner, f"{where}[{key!r}]")
            return result
        case DataclassNode(cls=cls, fields=fields):
            g = _expect_group(obj, "dataclass", where)
            expected = {fname for fname, _ in fields}
            if set(g.keys()) != expected:
                raise StorageFormatError(
                    f"{where}: dataclass fields {sorted(g.keys())} do not match "
                    f"{sorted(expected)}"
                )
            kwargs = {
                fname: read_node(g[fname], sub, f"{where}.{fname}")
                for fname, sub in fields
            }
            return cls(**kwargs)
        case CustomData(validator=validator):
            g = _expect_group(obj, "custom", where)
            nested = _read_nested(g, where)
            result = validator.func(nested)
            try:
                node.validate(result, where)
            except ValueTypeError as exc:
                raise SerializerContractError(
                    f"{where}: validator returned a wrong type: {exc}"
                ) from exc
            return result
        case CustomStatic():
            raise TypeError("custom static parameters are stored in the opts file")
        case _:
            raise TypeError(f"cannot read node {node!r}")


def _require_ordered(g: h5py.Group, where: str) -> None:
    flags = g.id.get_create_plist().get_link_creation_order()
    if not flags & h5py.h5p.CRT_ORDER_TRACKED:
        raise StorageFormatError(f"{where}: dict group does not track creation order")


def _read_nested(g: h5py.Group, where: str) -> dict[str, Any]:
    _require_ordered(g, where)
    out: dict[str, Any] = {}
    for name in g:
        child = g[name]
        here = f"{where}/{name}"
        if isinstance(child, h5py.Dataset):
            out[name] = _read_array(child, here)
        else:
            kind = _attr(child, "kind", here)
            if kind == "dict":
                out[name] = _read_nested(child, here)
            elif kind == "scalar":
                out[name] = _read_scalar(child, None, here)
            else:
                raise StorageFormatError(f"{here}: unexpected kind {kind!r}")
    return out
