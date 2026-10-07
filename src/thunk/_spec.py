"""Compile type annotations into immutable, strictly validating node trees."""

import dataclasses
import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import numpy as np

from . import _jax
from ._errors import SpecError, ValueTypeError
from ._markers import ROLE_MARKERS, Data, DataSerializer, DataValidator

ARRAY_KINDS = frozenset("biufcUS")
SCALAR_TYPES: tuple[type, ...] = (bool, int, float, str)

Json = Any


def qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or repr(obj)
    return f"{module}.{name}" if module else name


def _describe_type(tp: Any) -> str:
    return qualified_name(tp) if isinstance(tp, type) else repr(tp)


def _fail(path: str, expected: str, value: Any) -> typing.NoReturn:
    raise ValueTypeError(
        f"{path}: expected {expected}, got {type(value).__name__} ({value!r:.60})"
    )


class Node:
    """Base class: one compiled annotation."""

    @property
    def contains_array(self) -> bool:
        return False

    def validate(self, value: Any, path: str) -> None:
        raise NotImplementedError

    def describe(self) -> Json:
        raise NotImplementedError


@dataclass(frozen=True)
class Scalar(Node):
    type: type

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not self.type:
            _fail(path, self.type.__name__, value)

    def describe(self) -> Json:
        return {"kind": "scalar", "type": self.type.__name__}


@dataclass(frozen=True)
class NoneNode(Node):
    def validate(self, value: Any, path: str) -> None:
        if value is not None:
            _fail(path, "None", value)

    def describe(self) -> Json:
        return {"kind": "none"}


@dataclass(frozen=True)
class Array(Node):
    @property
    def contains_array(self) -> bool:
        return True

    def validate(self, value: Any, path: str) -> None:
        if not isinstance(value, np.ndarray):
            _fail(path, "numpy.ndarray", value)
        if value.dtype.kind not in ARRAY_KINDS:
            raise ValueTypeError(f"{path}: unsupported array dtype {value.dtype}")

    def describe(self) -> Json:
        return {"kind": "array"}


@dataclass(frozen=True)
class JaxArray(Node):
    @property
    def contains_array(self) -> bool:
        return True

    def validate(self, value: Any, path: str) -> None:
        _jax.validate(value, path)

    def describe(self) -> Json:
        return {"kind": "jax_array"}


@dataclass(frozen=True)
class LiteralNode(Node):
    values: tuple[Any, ...]

    def validate(self, value: Any, path: str) -> None:
        if not any(type(value) is type(v) and value == v for v in self.values):
            _fail(path, f"one of {list(self.values)!r}", value)

    def describe(self) -> Json:
        return {
            "kind": "literal",
            "values": [[type(v).__name__, v] for v in self.values],
        }


@dataclass(frozen=True)
class OptionalNode(Node):
    inner: Node

    @property
    def contains_array(self) -> bool:
        return self.inner.contains_array

    def validate(self, value: Any, path: str) -> None:
        if value is not None:
            self.inner.validate(value, path)

    def describe(self) -> Json:
        return {"kind": "optional", "inner": self.inner.describe()}


@dataclass(frozen=True)
class ListNode(Node):
    inner: Node

    @property
    def contains_array(self) -> bool:
        return self.inner.contains_array

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not list:
            _fail(path, "list", value)
        for i, item in enumerate(value):
            self.inner.validate(item, f"{path}[{i}]")

    def describe(self) -> Json:
        return {"kind": "list", "inner": self.inner.describe()}


@dataclass(frozen=True)
class VarTupleNode(Node):
    inner: Node

    @property
    def contains_array(self) -> bool:
        return self.inner.contains_array

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not tuple:
            _fail(path, "tuple", value)
        for i, item in enumerate(value):
            self.inner.validate(item, f"{path}[{i}]")

    def describe(self) -> Json:
        return {"kind": "var_tuple", "inner": self.inner.describe()}


@dataclass(frozen=True)
class FixedTupleNode(Node):
    items: tuple[Node, ...]

    @property
    def contains_array(self) -> bool:
        return any(n.contains_array for n in self.items)

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not tuple:
            _fail(path, "tuple", value)
        if len(value) != len(self.items):
            raise ValueTypeError(
                f"{path}: expected tuple of length {len(self.items)}, "
                f"got length {len(value)}"
            )
        for i, (node, item) in enumerate(zip(self.items, value, strict=True)):
            node.validate(item, f"{path}[{i}]")

    def describe(self) -> Json:
        return {"kind": "tuple", "items": [n.describe() for n in self.items]}


@dataclass(frozen=True)
class DictNode(Node):
    value: Node

    @property
    def contains_array(self) -> bool:
        return self.value.contains_array

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not dict:
            _fail(path, "dict", value)
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{path}.<key>", "str", key)
            self.value.validate(item, f"{path}[{key!r}]")

    def describe(self) -> Json:
        return {"kind": "dict", "value": self.value.describe()}


@dataclass(frozen=True)
class DataclassNode(Node):
    cls: type
    fields: tuple[tuple[str, Node], ...]

    @property
    def contains_array(self) -> bool:
        return any(n.contains_array for _, n in self.fields)

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not self.cls:
            _fail(path, self.cls.__name__, value)
        for name, node in self.fields:
            node.validate(getattr(value, name), f"{path}.{name}")

    def describe(self) -> Json:
        return {
            "kind": "dataclass",
            "type": qualified_name(self.cls),
            "fields": [[name, node.describe()] for name, node in self.fields],
        }


@dataclass(frozen=True)
class CustomData(Node):
    """A parameter stored through a user-supplied ``Data`` serializer pair."""

    annotation: Any
    serializer: DataSerializer
    validator: DataValidator

    @property
    def contains_array(self) -> bool:
        return True

    def validate(self, value: Any, path: str) -> None:
        if isinstance(self.annotation, type) and not isinstance(value, self.annotation):
            _fail(path, self.annotation.__name__, value)

    def describe(self) -> Json:
        return {
            "kind": "custom_data",
            "type": _describe_type(self.annotation),
            "serializer": qualified_name(self.serializer.func),
            "validator": qualified_name(self.validator.func),
        }


@dataclass(frozen=True)
class CustomStatic(Node):
    """A parameter stored through a pydantic serializer/validator pair."""

    annotation: Any
    serializer: Any
    validator: Any

    def validate(self, value: Any, path: str) -> None:
        if isinstance(self.annotation, type) and not isinstance(value, self.annotation):
            _fail(path, self.annotation.__name__, value)

    def describe(self) -> Json:
        return {
            "kind": "custom_static",
            "type": _describe_type(self.annotation),
            "serializer": qualified_name(self.serializer.func),
            "validator": qualified_name(self.validator.func),
        }


def contains_role_marker(metadata: tuple[Any, ...]) -> bool:
    return any(isinstance(m, ROLE_MARKERS) for m in metadata)


def compile_annotation(tp: Any) -> Node:
    """Compile ``tp`` into a ``Node``; raise ``SpecError`` if unsupported."""
    return _compile(tp, (), "<annotation>")


def _compile(tp: Any, stack: tuple[Any, ...], where: str) -> Node:
    # PEP 695 aliases
    if isinstance(tp, typing.TypeAliasType):
        if tp in stack:
            raise SpecError(f"{where}: recursive type {tp.__name__} is not supported")
        return _compile(tp.__value__, (*stack, tp), where)

    origin = get_origin(tp)

    if isinstance(origin, typing.TypeAliasType):
        # e.g. ``npt.NDArray[np.float64]``: subscripting a PEP 695 alias yields
        # a GenericAlias whose origin is the alias itself, not the aliased type
        if origin in stack:
            raise SpecError(
                f"{where}: recursive type {origin.__name__} is not supported"
            )
        try:
            resolved = origin.__value__[get_args(tp)]
        except TypeError as exc:
            raise SpecError(f"{where}: unsupported annotation {tp!r} ({exc})") from exc
        return _compile(resolved, (*stack, origin), where)

    if origin is Annotated:
        meta = tp.__metadata__
        if contains_role_marker(meta):
            raise SpecError(
                f"{where}: role markers (Data/Static/Skip) are only allowed on a "
                "parameter's top-level annotation"
            )
        return _compile(tp.__origin__, stack, where)

    if tp is None or tp is type(None):
        return NoneNode()
    if tp in SCALAR_TYPES:
        return Scalar(tp)
    if tp is np.ndarray or origin is np.ndarray:
        return Array()
    if _jax.is_annotation(tp):
        return JaxArray()
    if tp is typing.Any:
        raise SpecError(f"{where}: Any is not supported")

    if origin is Literal:
        values = get_args(tp)
        for v in values:
            if type(v) not in (bool, int, str):
                raise SpecError(
                    f"{where}: Literal values must be bool, int or str, got {v!r}"
                )
        return LiteralNode(tuple(values))

    if origin in (Union, types.UnionType):
        args = get_args(tp)
        rest = [a for a in args if a is not type(None)]
        if len(args) == 2 and len(rest) == 1:
            return OptionalNode(_compile(rest[0], stack, where))
        raise SpecError(f"{where}: only `T | None` unions are supported, got {tp!r}")

    if origin is list:
        (inner,) = _args(tp, 1, where)
        return ListNode(_compile(inner, stack, f"{where}[]"))

    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return VarTupleNode(_compile(args[0], stack, f"{where}[]"))
        if Ellipsis in args:
            raise SpecError(f"{where}: malformed tuple annotation {tp!r}")
        return FixedTupleNode(
            tuple(_compile(a, stack, f"{where}[{i}]") for i, a in enumerate(args))
        )

    if origin is dict:
        key, value = _args(tp, 2, where)
        if key is not str:
            raise SpecError(f"{where}: dict keys must be str, got {key!r}")
        return DictNode(_compile(value, stack, f"{where}[]"))

    if tp in (list, dict, tuple):
        raise SpecError(f"{where}: bare {tp.__name__} is not supported; parametrize it")

    if origin is None and isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _compile_dataclass(tp, stack)

    raise SpecError(f"{where}: unsupported annotation {tp!r}")


def _args(tp: Any, n: int, where: str) -> tuple[Any, ...]:
    args = get_args(tp)
    if len(args) != n:
        raise SpecError(f"{where}: unsupported annotation {tp!r}")
    return args


def _compile_dataclass(cls: Any, stack: tuple[Any, ...]) -> Node:
    name = qualified_name(cls)
    if cls in stack:
        raise SpecError(f"{name}: recursive dataclasses are not supported")
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception as exc:
        raise SpecError(f"{name}: cannot resolve field annotations: {exc}") from exc

    fields: list[tuple[str, Node]] = []
    for field in dataclasses.fields(cls):
        where = f"{name}.{field.name}"
        if not field.init:
            raise SpecError(f"{where}: fields with init=False are not supported")
        fields.append((field.name, _compile(hints[field.name], (*stack, cls), where)))
    for key, hint in hints.items():
        if isinstance(hint, dataclasses.InitVar):
            raise SpecError(f"{name}.{key}: InitVar fields are not supported")
    return DataclassNode(cls, tuple(fields))


def wrap_custom_data(annotation: Any, marker: Data) -> CustomData:
    assert marker.serializer is not None and marker.validator is not None
    return CustomData(annotation, marker.serializer, marker.validator)
