"""Node-driven HDF5 writer and reader."""

import json
import math
import re
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import _jax
from ._errors import StorageFormatError
from ._spec import Node

STORAGE_VERSION = 2
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}

# -- file envelope -------------------------------------------------------


def write_envelope(
    f: h5py.File, kind: str, fingerprints: Mapping[str, str], **extra: str | int
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
        if not isinstance(found, str) or found != kind:
            raise StorageFormatError(
                f"{path}: expected a thunk {kind!r} file, found {found!r}"
            )
        version = _attr(f, "storage_version", path)
        if (
            not isinstance(version, Integral)
            or isinstance(version, bool)
            or version != STORAGE_VERSION
        ):
            raise StorageFormatError(
                f"{path}: unsupported storage_version {version!r} "
                f"(this thunk reads {STORAGE_VERSION})"
            )
        try:
            fps = json.loads(_attr(f, "fingerprints", path))
        except (ValueError, TypeError) as exc:
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
    if ds.shape is None:
        raise StorageFormatError(f"{where}: null array dataset")
    data = np.asarray(ds[()])
    if "numpy_dtype" in ds.attrs:
        dtype = np.dtype(_attr(ds, "numpy_dtype", where))
        if dtype.kind != "U" or data.dtype != np.dtype("<u4") or data.ndim < 1:
            raise StorageFormatError(f"{where}: malformed unicode array")
        if data.shape[-1] < 1 or data.shape[-1] != max(dtype.itemsize // 4, 1):
            raise StorageFormatError(f"{where}: malformed unicode width")
        shape = data.shape[:-1]
        nchar = data.shape[-1]
        native = np.asarray(data, order="C").reshape(-1).view(f"<U{nchar}")
        return native.astype(dtype).reshape(shape)
    if data.dtype.kind not in "biufcS":
        raise StorageFormatError(f"{where}: unsupported dtype {data.dtype}")
    return data


def _group(
    parent: h5py.Group, name: str, kind: str, *, track_order: bool = False
) -> h5py.Group:
    g = parent.create_group(name, track_order=track_order)
    g.attrs["kind"] = kind
    return g


def _write_none(parent: h5py.Group, name: str) -> None:
    _group(parent, name, "none")


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


def write_node(parent: h5py.Group, name: str, node: Node, value: Any) -> None:
    kind = node.kind
    if kind in ("scalar", "nonfinite"):
        _write_scalar(parent, name, value)
    elif kind == "none":
        _write_none(parent, name)
    elif kind == "array":
        _write_array(parent, name, value)
    elif kind == "jax_array":
        data, implementation = _jax.to_host(value, name)
        ds = parent.create_dataset(name, data=data)
        ds.attrs["jax_kind"] = "prng_key" if implementation else "array"
        if implementation is not None:
            ds.attrs["jax_impl"] = implementation
    else:
        g = _group(parent, name, kind, track_order=True)
        if node.type is not None:
            g.attrs["type"] = node.type
        for i, (key, sub) in enumerate(node.children):
            sequence = kind in ("list", "tuple")
            write_node(
                g,
                key if sequence else escape_key(key),
                sub,
                value[i] if sequence else value[key],
            )


def read_node(obj: Any, where: str) -> tuple[Node, Any]:
    """Decode stored tags only. Never import stored dataclass names."""
    if isinstance(obj, h5py.Dataset):
        if "jax_kind" in obj.attrs:
            kind = _attr(obj, "jax_kind", where)
            if not isinstance(kind, str) or kind not in ("array", "prng_key"):
                raise StorageFormatError(f"{where}: unsupported jax_kind")
            impl = _attr(obj, "jax_impl", where) if kind == "prng_key" else None
            if impl is not None and not isinstance(impl, str):
                raise StorageFormatError(f"{where}: malformed PRNG implementation")
            if obj.shape is None:
                raise StorageFormatError(f"{where}: invalid JAX shape")
            return Node("jax_array"), _jax.from_host(
                np.asarray(obj[()]), kind, impl, where
            )
        return Node("array"), _read_array(obj, where)
    kind = _attr(obj, "kind", where)
    if kind == "scalar":
        if len(obj):
            raise StorageFormatError(f"{where}: scalar has children")
        value = _read_scalar(obj, None, where)
        from ._spec import infer

        return infer(value, where), value
    if kind == "none":
        if len(obj):
            raise StorageFormatError(f"{where}: none has children")
        return Node("none"), None
    if kind in ("list", "tuple"):
        children = [
            read_node(c, f"{where}[{i}]") for i, c in enumerate(_indexed(obj, where))
        ]
        values = [v for _, v in children]
        return Node(
            kind, tuple((str(i), n) for i, (n, _) in enumerate(children))
        ), tuple(values) if kind == "tuple" else values
    if kind in ("dict", "dataclass"):
        _require_ordered(obj, where)
        children, values = [], {}
        for name in obj:
            key = unescape_key(name, where)
            if escape_key(key) != name or key in values:
                raise StorageFormatError(f"{where}: invalid key {name!r}")
            n, v = read_node(obj[name], f"{where}[{key!r}]")
            children.append((key, n))
            values[key] = v
        tp = _attr(obj, "type", where) if kind == "dataclass" else None
        return Node(kind, tuple(children), tp), values
    raise StorageFormatError(f"{where}: unknown node kind {kind!r}")


def _require_ordered(g: h5py.Group, where: str) -> None:
    flags = g.id.get_create_plist().get_link_creation_order()
    if not flags & h5py.h5p.CRT_ORDER_TRACKED:
        raise StorageFormatError(f"{where}: dict group does not track creation order")
