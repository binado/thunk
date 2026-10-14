"""Concrete runtime representations, encoding, and optional reconstruction hints."""

import dataclasses
import math
import sys
import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin

import numpy as np

from . import _jax
from ._errors import ReconstructionError, SerializerContractError, ValueTypeError

ARRAY_KINDS = frozenset("biufcUS")


def qualified_name(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"


@dataclass(frozen=True)
class Node:
    kind: str
    children: tuple[tuple[str, "Node"], ...] = ()
    type: str | None = None

    @property
    def contains_array(self) -> bool:
        return self.kind in ("array", "jax_array", "nonfinite") or any(
            n.contains_array for _, n in self.children
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "type": self.type,
            "children": [[k, n.describe()] for k, n in self.children],
        }

    @classmethod
    def from_description(cls, raw: Any) -> "Node":
        from ._errors import StorageFormatError

        try:
            if not isinstance(raw, dict) or set(raw) != {"kind", "type", "children"}:
                raise ValueError("invalid representation")
            kind, tp, children = raw["kind"], raw["type"], raw["children"]
            if kind not in {
                "scalar",
                "none",
                "nonfinite",
                "array",
                "jax_array",
                "list",
                "tuple",
                "dict",
                "dataclass",
            }:
                raise ValueError("unknown representation kind")
            if tp is not None and not isinstance(tp, str):
                raise ValueError("invalid representation type")
            if not isinstance(children, list):
                raise ValueError("invalid children")
            pairs = []
            for pair in children:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                ):
                    raise ValueError("invalid child")
                pairs.append((pair[0], cls.from_description(pair[1])))
            if len({k for k, _ in pairs}) != len(pairs):
                raise ValueError("duplicate children")
            if kind in ("list", "tuple") and [k for k, _ in pairs] != [
                str(i) for i in range(len(pairs))
            ]:
                raise ValueError("invalid sequence")
            if kind not in ("list", "tuple", "dict", "dataclass") and pairs:
                raise ValueError("unexpected children")
            if kind == "scalar" and tp not in ("bool", "int", "float", "str"):
                raise ValueError("invalid scalar type")
            if kind == "dataclass" and (not tp or not isinstance(tp, str)):
                raise ValueError("invalid dataclass name")
            if kind == "nonfinite" and tp != "float":
                raise ValueError("invalid nonfinite type")
            if kind not in ("scalar", "dataclass", "nonfinite") and tp is not None:
                raise ValueError("unexpected representation type")
            return cls(kind, tuple(pairs), tp)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise StorageFormatError(f"invalid runtime representation: {exc}") from exc


def infer(value: Any, path: str, stack: tuple[int, ...] = ()) -> Node:
    """Inspect values without copying arrays, transferring devices, or running codecs."""
    if value is None:
        return Node("none")
    if type(value) in (bool, int, float, str):
        kind = (
            "nonfinite"
            if type(value) is float and not math.isfinite(value)
            else "scalar"
        )
        return Node(kind, type=type(value).__name__)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in ARRAY_KINDS:
            raise ValueTypeError(f"{path}: unsupported array dtype {value.dtype}")
        return Node("array")
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(value, jax.Array):
        _jax.validate(value, path)
        return Node("jax_array")
    if id(value) in stack:
        raise ValueTypeError(f"{path}: cycle in runtime value")
    stack = (*stack, id(value))
    if type(value) in (list, tuple):
        return Node(
            type(value).__name__,
            tuple(
                (str(i), infer(v, f"{path}[{i}]", stack)) for i, v in enumerate(value)
            ),
        )
    if type(value) is dict:
        children = []
        for k, v in value.items():
            if type(k) is not str:
                raise ValueTypeError(f"{path}.<key>: expected str, got {k!r}")
            children.append((k, infer(v, f"{path}[{k!r}]", stack)))
        return Node("dict", tuple(children))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        for field in value.__dataclass_fields__.values():
            if field._field_type is getattr(dataclasses, "_FIELD_INITVAR"):
                raise ValueTypeError(
                    f"{path}.{field.name}: InitVar fields are not supported"
                )
        children = []
        for field in fields:
            if not field.init:
                raise ValueTypeError(
                    f"{path}.{field.name}: init=False fields are not supported"
                )
            children.append(
                (
                    field.name,
                    infer(getattr(value, field.name), f"{path}.{field.name}", stack),
                )
            )
        return Node("dataclass", tuple(children), qualified_name(type(value)))
    raise ValueTypeError(f"{path}: unsupported value type {type(value).__name__}")


def encode(node: Node, value: Any, *, json_mode: bool = False) -> Any:
    if node.kind in ("list", "tuple"):
        items = [
            encode(n, v, json_mode=json_mode)
            for (_, n), v in zip(node.children, value, strict=True)
        ]
        return tuple(items) if node.kind == "tuple" and not json_mode else items
    if node.kind in ("dict", "dataclass"):
        return {
            k: encode(
                n,
                getattr(value, k) if node.kind == "dataclass" else value[k],
                json_mode=json_mode,
            )
            for k, n in node.children
        }
    return value


def validate_payload(
    node: Node, value: Any, path: str, *, json_mode: bool = False
) -> None:
    """Check a decoded payload against its descriptor, without user reconstruction."""
    from ._errors import StorageFormatError

    try:
        if node.kind in ("list", "tuple"):
            expected = list if json_mode or node.kind == "list" else tuple
            if type(value) is not expected or len(value) != len(node.children):
                raise ValueError("sequence shape differs")
            for (k, sub), v in zip(node.children, value, strict=True):
                validate_payload(sub, v, f"{path}[{k}]", json_mode=json_mode)
        elif node.kind in ("dict", "dataclass"):
            if type(value) is not dict or list(value) != [k for k, _ in node.children]:
                raise ValueError("fields or field order differ")
            for k, sub in node.children:
                validate_payload(sub, value[k], f"{path}[{k!r}]", json_mode=json_mode)
        elif infer(value, path) != node:
            raise ValueError("scalar or array representation differs")
    except (ValueError, TypeError) as exc:
        raise StorageFormatError(f"{path}: invalid payload: {exc}") from exc


def reconstruct(value: Any, hint: Any, path: str, *, hdf5: bool = False) -> Any:
    """Apply compatible structural hints; never coerce or validate scalar values."""
    seen_aliases = set()
    while isinstance(hint, typing.TypeAliasType):
        if id(hint) in seen_aliases:
            return value
        seen_aliases.add(id(hint))
        hint = hint.__value__
    origin, args = get_origin(hint), get_args(hint)
    if origin is Annotated:
        return reconstruct(value, args[0], path, hdf5=hdf5)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(args) == 2 and len(rest) == 1 and value is not None:
            return reconstruct(value, rest[0], path, hdf5=hdf5)
        return value
    if origin in (list, tuple) or hint is list or hint is tuple:
        if not isinstance(value, (list, tuple)):
            return value
        is_tuple = origin is tuple or hint is tuple
        if (
            is_tuple
            and hint is not tuple
            and not (len(args) == 2 and args[1] is Ellipsis)
            and len(args) != len(value)
        ):
            return value
        hints = (
            [args[0]] * len(value)
            if args and (not is_tuple or (len(args) == 2 and args[1] is Ellipsis))
            else list(args)
        )
        result = [
            reconstruct(
                v, hints[i] if i < len(hints) else Any, f"{path}[{i}]", hdf5=hdf5
            )
            for i, v in enumerate(value)
        ]
        return (
            tuple(result)
            if (isinstance(value, tuple) if hdf5 else is_tuple)
            else result
        )
    if origin is dict and isinstance(value, dict):
        child = args[1] if len(args) == 2 else Any
        return {
            k: reconstruct(v, child, f"{path}[{k!r}]", hdf5=hdf5)
            for k, v in value.items()
        }
    if (
        isinstance(hint, type)
        and dataclasses.is_dataclass(hint)
        and isinstance(value, dict)
    ):
        fields = {f.name: f for f in dataclasses.fields(hint) if f.init}
        if set(value) - set(fields) or any(
            k not in value
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            for k, f in fields.items()
        ):
            return value
        try:
            hints = typing.get_type_hints(hint, include_extras=True)
        except Exception:
            hints = getattr(hint, "__annotations__", {})
        kwargs = {
            k: reconstruct(v, hints.get(k, Any), f"{path}.{k}", hdf5=hdf5)
            for k, v in value.items()
        }
        try:
            return hint(**kwargs)
        except Exception as exc:
            raise ReconstructionError(
                f"{path}: {qualified_name(hint)} constructor failed: {exc}"
            ) from exc
    return value


def codec_identity(marker: Any) -> dict[str, str] | None:
    if marker is None or getattr(marker, "serializer", None) is None:
        return None
    return {
        "kind": type(marker).__name__,
        "serializer": qualified_name(marker.serializer.func),
        "deserializer": qualified_name(marker.validator.func),
    }


def run_codec(marker: Any, value: Any, path: str, *, decode: bool = False) -> Any:
    from pydantic import TypeAdapter

    from ._markers import Static

    try:
        if isinstance(marker, Static):
            # Any deliberately disables ordinary annotation validation. Pydantic
            # still supplies its supported serialization/validation info objects.
            if decode:
                return TypeAdapter(Annotated[Any, marker.validator]).validate_python(
                    value
                )
            return TypeAdapter(Annotated[Any, marker.serializer]).dump_python(
                value, mode="python", warnings="error"
            )
        return (marker.validator if decode else marker.serializer).func(value)
    except Exception as exc:
        operation = "deserializer" if decode else "serializer"
        raise SerializerContractError(f"{path}: {operation} failed: {exc}") from exc


def validate_custom_data(value: Any, path: str, stack: tuple[int, ...] = ()) -> None:
    from collections.abc import Mapping

    if not isinstance(value, Mapping) or id(value) in stack:
        raise SerializerContractError(
            f"{path}: serializer must return an acyclic nested dict"
        )
    for k, v in value.items():
        here = f"{path}[{k!r}]"
        if not isinstance(k, str) or not k or "/" in k or k in (".", "..") or "\0" in k:
            raise SerializerContractError(f"{here}: invalid key")
        if isinstance(v, Mapping):
            validate_custom_data(v, here, (*stack, id(value)))
        elif type(v) not in (bool, int, float, str) and not isinstance(v, np.ndarray):
            raise SerializerContractError(f"{here}: unsupported serializer value")
        else:
            try:
                infer(v, here)
            except ValueTypeError as exc:
                raise SerializerContractError(str(exc)) from exc
