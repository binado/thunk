"""Schema fingerprints and content digests."""

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _jax
from ._spec import Node


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(name: str, role: str, node: Node) -> str:
    return _sha256(
        canonical_json({"name": name, "role": role, "spec": node.describe()})
    )


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
                self.blob(
                    b"f", struct.pack(">d", math.nan if math.isnan(value) else value)
                )
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

    def feed(self, node: Node, value: Any) -> None:
        self.blob(b"t", node.kind.encode())
        if node.kind == "array":
            self.array(value)
        elif node.kind == "jax_array":
            data, implementation = _jax.to_host(value, "content digest")
            self.scalar(implementation)
            self.array(data)
        elif node.kind in ("dict", "dataclass", "list", "tuple"):
            for i, (key, sub) in enumerate(node.children):
                self.scalar(key)
                self.feed(
                    sub, value[i] if node.kind in ("list", "tuple") else value[key]
                )
        else:
            self.scalar(value)

    def hexdigest(self) -> str:
        return self.h.hexdigest()


def group_digest(nodes: Mapping[str, Node], values: Mapping[str, Any]) -> str:
    hasher = _Hasher()
    for name, node in nodes.items():
        hasher.blob(b"p", name.encode())
        hasher.feed(node, values[name])
    return hasher.hexdigest()


def json_digest(encoded: Mapping[str, Any]) -> str:
    """Digest of an opts group's JSON encoding (insertion order is significant)."""
    return _sha256(json.dumps(encoded, separators=(",", ":"), allow_nan=False))


DIGEST_VERSION = 2


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
