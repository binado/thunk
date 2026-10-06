"""Signature compilation, role resolution, and argument binding."""

import inspect
import sys
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_origin

from ._errors import OutputCodecError, SpecError
from ._markers import ROLE_MARKERS, Data, Skip, Static
from ._spec import (
    CustomData,
    CustomStatic,
    Node,
    NoneNode,
    compile_annotation,
)

Role = Literal["data", "static", "skip"]
EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class Param:
    name: str
    kind: inspect._ParameterKind
    default: Any
    role: Role
    node: Node | None
    # Static only: the annotation handed to pydantic (base type + non-role metadata
    # + custom serializer/validator).
    pydantic_annotation: Any = None

    @property
    def has_default(self) -> bool:
        return self.default is not EMPTY


def _blank() -> None: ...


def _namespace(target: Any) -> dict[str, Any]:
    target = inspect.unwrap(target) if callable(target) else target
    if isinstance(target, type):
        module = sys.modules.get(target.__module__)
        return vars(module) if module else {}
    return getattr(target, "__globals__", {})


def _resolve(raw: Any, globalns: dict[str, Any]) -> Any:
    """Resolve one (possibly string) annotation, keeping ``Annotated`` extras."""
    stub = types.FunctionType(_blank.__code__, globalns)
    stub.__annotations__ = {"x": raw}
    return typing.get_type_hints(stub, include_extras=True)["x"]


def _split_roles(annotation: Any) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Return ``(base, role_markers, other_metadata)`` for a top-level annotation."""
    if get_origin(annotation) is not Annotated:
        return annotation, (), ()
    meta = annotation.__metadata__
    roles = tuple(m for m in meta if isinstance(m, ROLE_MARKERS))
    others = tuple(m for m in meta if not isinstance(m, ROLE_MARKERS))
    return annotation.__origin__, roles, others


def _raw_skip(raw: Any) -> bool:
    """Detect ``Annotated[<unresolvable>, Skip()]`` without resolving the base."""
    return get_origin(raw) is Annotated and any(
        isinstance(m, Skip) for m in raw.__metadata__
    )


def _compile_param(p: inspect.Parameter, globalns: dict[str, Any], owner: str) -> Param:
    name = p.name
    raw = p.annotation
    if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
        raise SpecError(f"{owner}: *args/**kwargs parameter {name!r} is not supported")

    if raw is EMPTY:
        raise SpecError(
            f"{owner}: parameter {name!r} needs a type annotation "
            "(or mark it Annotated[..., thunk.Skip()])"
        )
    try:
        annotation = _resolve(raw, globalns)
    except Exception as exc:
        if _raw_skip(raw):
            return Param(name, p.kind, p.default, "skip", None)
        raise SpecError(
            f"{owner}: cannot resolve annotation of parameter {name!r}: {exc}"
        ) from exc

    base, roles, others = _split_roles(annotation)
    if len(roles) > 1:
        raise SpecError(f"{owner}: parameter {name!r} has more than one role marker")
    marker = roles[0] if roles else None

    if isinstance(marker, Skip):
        return Param(name, p.kind, p.default, "skip", None)

    try:
        if isinstance(marker, Data) and marker.serializer is not None:
            node: Node = CustomData(base, marker.serializer, marker.validator)
            return Param(name, p.kind, p.default, "data", node)
        if isinstance(marker, Static) and marker.serializer is not None:
            node = CustomStatic(base, marker.serializer, marker.validator)
            pyd = Annotated[base, *others, marker.serializer, marker.validator]
            return Param(name, p.kind, p.default, "static", node, pyd)

        node = compile_annotation(base)
    except SpecError as exc:
        raise SpecError(f"{owner}: parameter {name!r}: {exc}") from exc

    if isinstance(marker, Static):
        role: Role = "static"
        if node.contains_array:
            raise SpecError(
                f"{owner}: parameter {name!r} is marked Static but contains arrays"
            )
    elif isinstance(marker, Data):
        role = "data"
    else:
        role = "data" if node.contains_array else "static"

    pyd = Annotated[base, *others] if others else base
    return Param(name, p.kind, p.default, role, node, pyd if role == "static" else None)


class CallSpec:
    """Compiled view of a callable's signature, split by argument role."""

    def __init__(self, function: Callable[..., Any]) -> None:
        if not callable(function):
            raise SpecError(f"thunk.fn expects a callable, got {function!r}")
        self.function = function
        self.name = getattr(function, "__qualname__", repr(function))
        try:
            self.signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise SpecError(f"{self.name}: cannot inspect signature: {exc}") from exc

        target = _hint_target(function)
        globalns = _namespace(target)
        self.params: tuple[Param, ...] = tuple(
            _compile_param(p, globalns, self.name)
            for p in self.signature.parameters.values()
        )
        self.by_name = {p.name: p for p in self.params}
        self.data = tuple(p for p in self.params if p.role == "data")
        self.static = tuple(p for p in self.params if p.role == "static")
        self.skip = tuple(p for p in self.params if p.role == "skip")

        self.output_node: Node | None = None
        self.output_error: Exception | None = None
        self._compile_output(globalns)

    def _compile_output(self, globalns: dict[str, Any]) -> None:
        raw = self.signature.return_annotation
        if raw is inspect.Signature.empty:
            self.output_error = SpecError(f"{self.name}: no return annotation")
            return
        try:
            tp = _resolve(raw, globalns)
            self.output_node = NoneNode() if tp is None else compile_annotation(tp)
        except Exception as exc:
            self.output_error = exc

    def require_output(self) -> Node:
        if self.output_node is None:
            raise OutputCodecError(
                f"{self.name}: cannot derive an output codec: {self.output_error}"
            ) from self.output_error
        return self.output_node

    # -- binding ---------------------------------------------------------

    def bind(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        inputs = {p.name: values[p.name] for p in self.data}
        opts = {p.name: values[p.name] for p in self.static}
        return inputs, opts

    def check_names(
        self, group: Mapping[str, Any], params: tuple[Param, ...], label: str
    ) -> None:
        expected = {p.name for p in params}
        unknown = [k for k in group if k not in expected]
        missing = [n for n in expected if n not in group]
        if unknown or missing:
            raise TypeError(
                f"{self.name}: {label} mismatch"
                + (f"; unknown: {unknown}" if unknown else "")
                + (f"; missing: {sorted(missing)}" if missing else "")
            )

    def merge(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        skipped: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        self.check_names(inputs, self.data, "inputs")
        self.check_names(opts, self.static, "opts")
        skipped = skipped or {}
        skip_names = {p.name for p in self.skip}
        bad = [k for k in skipped if k not in skip_names]
        if bad:
            raise TypeError(
                f"{self.name}: `skipped` only accepts Skip parameters, got {bad}"
            )
        merged: dict[str, Any] = {}
        for p in self.params:
            if p.role == "data":
                merged[p.name] = inputs[p.name]
            elif p.role == "static":
                merged[p.name] = opts[p.name]
            elif p.name in skipped:
                merged[p.name] = skipped[p.name]
            elif p.has_default:
                merged[p.name] = p.default
            else:
                raise TypeError(
                    f"{self.name}: missing required skipped parameter {p.name!r}"
                )
        return merged

    def to_call(
        self, values: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for p in self.params:
            if p.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(values[p.name])
            else:
                kwargs[p.name] = values[p.name]
        return tuple(args), kwargs


def _hint_target(function: Callable[..., Any]) -> Any:
    if inspect.isroutine(function) or isinstance(function, type):
        return function
    return type(function).__call__
