"""Schema fingerprints and content digests."""

import hashlib
import json
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _jax
from ._signature import Param
from ._spec import (
    Array,
    CustomData,
    CustomStatic,
    DataclassNode,
    DictNode,
    FixedTupleNode,
    JaxArray,
    ListNode,
    LiteralNode,
    Node,
    NoneNode,
    OptionalNode,
    Scalar,
    VarTupleNode,
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(name: str, role: str, node: Node) -> str:
    return _sha256(
        canonical_json({"name": name, "role": role, "spec": node.describe()})
    )


def param_fingerprint(param: Param) -> str:
    assert param.node is not None
    return fingerprint(param.name, param.role, param.node)


def output_fingerprint(node: Node) -> str:
    return fingerprint("output", "output", node)


class _Hasher:
    """Length-framed, type-tagged feed into SHA-256."""

    def __init__(self) -> None:
        self.h = hashlib.sha256()

    def tag(self, tag: bytes) -> None:
        self.h.update(tag)

    def blob(self, tag: bytes, data: bytes) -> None:
        self.h.update(tag + struct.pack(">Q", len(data)) + data)

    def scalar(self, value: Any) -> None:
        match value:
            case bool():
                self.blob(b"b", b"1" if value else b"0")
            case int():
                self.blob(b"i", str(value).encode())
            case float():
                self.blob(b"f", struct.pack(">d", value))
            case str():
                self.blob(b"s", value.encode())
            case None:
                self.tag(b"N")
            case _:
                raise TypeError(f"cannot digest scalar {value!r}")

    def array(self, value: np.ndarray) -> None:
        arr = np.asarray(value, order="C")
        dtype = arr.dtype.newbyteorder("<")
        arr = arr.astype(dtype, copy=False)
        self.blob(b"a", f"{dtype.str}|{arr.shape}".encode())
        self.blob(b"d", arr.tobytes())

    def nested(self, value: Any) -> None:
        """Digest an arbitrary (already validated) serializer-output tree."""
        if isinstance(value, Mapping):
            self.blob(b"{", struct.pack(">Q", len(value)))
            for key, item in value.items():
                self.scalar(key)
                self.nested(item)
        elif isinstance(value, np.ndarray):
            self.array(value)
        else:
            self.scalar(value)

    def feed(self, node: Node, value: Any, *, serialized: bool = False) -> None:
        match node:
            case Scalar() | LiteralNode():
                self.scalar(value)
            case NoneNode():
                self.tag(b"N")
            case Array():
                self.array(value)
            case JaxArray():
                data, implementation = _jax.to_host(value, "content digest")
                if implementation is None:
                    self.tag(b"J")
                else:
                    self.blob(b"K", implementation.encode())
                self.array(data)
            case OptionalNode(inner=inner):
                if value is None:
                    self.tag(b"N")
                else:
                    self.tag(b"S")
                    self.feed(inner, value, serialized=serialized)
            case ListNode(inner=inner) | VarTupleNode(inner=inner):
                self.blob(b"[", struct.pack(">Q", len(value)))
                for item in value:
                    self.feed(inner, item, serialized=serialized)
            case FixedTupleNode(items=items):
                self.tag(b"(")
                for sub, item in zip(items, value, strict=True):
                    self.feed(sub, item, serialized=serialized)
            case DictNode(value=inner):
                self.blob(b"{", struct.pack(">Q", len(value)))
                for key, item in value.items():
                    self.scalar(key)
                    self.feed(inner, item, serialized=serialized)
            case DataclassNode(fields=fields):
                self.tag(b"D")
                for name, sub in fields:
                    self.feed(
                        sub,
                        value[name] if serialized else getattr(value, name),
                        serialized=serialized,
                    )
            case CustomData(serializer=serializer):
                self.tag(b"C")
                self.nested(value if serialized else serializer.func(value))
            case CustomStatic():
                raise TypeError("custom static values are digested through JSON")
            case _:
                raise TypeError(f"cannot digest node {node!r}")

    def hexdigest(self) -> str:
        return self.h.hexdigest()


def group_digest(
    params: tuple[Param, ...], values: Mapping[str, Any], *, serialized: bool = False
) -> str:
    """Content digest of a group of ``Data`` parameter values."""
    hasher = _Hasher()
    for p in params:
        assert p.node is not None
        hasher.blob(b"p", p.name.encode())
        hasher.feed(p.node, values[p.name], serialized=serialized)
    return hasher.hexdigest()


def json_digest(encoded: Mapping[str, Any]) -> str:
    """Digest of an opts group's JSON encoding (insertion order is significant)."""
    return _sha256(json.dumps(encoded, separators=(",", ":"), allow_nan=False))


DIGEST_VERSION = 1


def combined_digest(
    fingerprints: Mapping[str, Mapping[str, str]],
    inputs_digest: str,
    opts_digest: str,
) -> str:
    return _sha256(
        canonical_json(
            {
                "digest_version": DIGEST_VERSION,
                "fingerprints": fingerprints,
                "inputs_digest": inputs_digest,
                "opts_digest": opts_digest,
            }
        )
    )
