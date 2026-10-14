"""Signature inspection independent of runtime storage inference."""

import ast
import dataclasses
import functools
import inspect
import sys
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from ._errors import SpecError, ValueTypeError
from ._markers import ROLE_MARKERS, Data, Skip, Static
from ._spec import infer

Role = Literal["data", "static", "skip"]
EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class Param:
    name: str
    kind: inspect._ParameterKind
    default: Any
    role: Role | None
    hint: Any = Any
    marker: Any = None

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


def _annotation(raw: Any, ns: dict[str, Any]) -> Any:
    if raw is EMPTY:
        return Any
    try:
        return _resolve(raw, ns)
    except Exception:
        if get_origin(raw) is Annotated:
            return raw
        if not isinstance(raw, str):
            return raw
        try:
            expr = ast.parse(raw, mode="eval").body
        except SyntaxError:
            return raw
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return _annotation(expr.value, ns)
        # Inspect Annotated metadata independently of unresolved ordinary types.
        # Walking the base as well preserves nested-marker errors even when
        # resolving its ordinary forward references failed.
        result = raw
        for sub in ast.walk(expr):
            if not isinstance(sub, ast.Subscript) or not isinstance(
                sub.slice, ast.Tuple
            ):
                continue
            try:
                outer = eval(
                    compile(ast.Expression(sub.value), "<annotation>", "eval"), ns
                )
            except Exception:
                continue
            if outer is not Annotated:
                continue
            try:
                meta = [
                    eval(compile(ast.Expression(e), "<annotation>", "eval"), ns)
                    for e in sub.slice.elts[1:]
                ]
            except Exception as exc:
                raise SpecError(f"cannot resolve persistence metadata: {raw}") from exc
            if sub is expr:
                result = Annotated[Any, *meta]
            elif any(isinstance(m, ROLE_MARKERS) for m in meta):
                raise SpecError(
                    "role markers are only allowed on a top-level annotation"
                )
        return result


def _nested(hint: Any, seen: tuple[int, ...] = ()) -> None:
    if id(hint) in seen:
        return
    seen = (*seen, id(hint))
    if isinstance(hint, typing.TypeAliasType):
        _nested(hint.__value__, seen)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        for raw in getattr(hint, "__annotations__", {}).values():
            _nested(_annotation(raw, _namespace(hint)), seen)
    if get_origin(hint) is Annotated and any(
        isinstance(m, ROLE_MARKERS) for m in hint.__metadata__
    ):
        raise SpecError("role markers are only allowed on a top-level annotation")
    for arg in get_args(hint):
        _nested(arg, seen)


def _compile_param(p: inspect.Parameter, globalns: dict[str, Any], owner: str) -> Param:
    if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
        raise SpecError(
            f"{owner}: *args/**kwargs parameter {p.name!r} is not supported"
        )
    annotation = _annotation(p.annotation, globalns)
    seen_aliases = set()
    while (
        isinstance(annotation, typing.TypeAliasType)
        and id(annotation) not in seen_aliases
    ):
        seen_aliases.add(id(annotation))
        annotation = annotation.__value__
    hint, roles, _ = _split_roles(annotation)
    if len(roles) > 1:
        raise SpecError(f"{owner}: parameter {p.name!r} has more than one role marker")
    _nested(hint)
    marker = roles[0] if roles else None
    role: Role | None = (
        "data"
        if isinstance(marker, Data)
        else "static"
        if isinstance(marker, Static)
        else "skip"
        if isinstance(marker, Skip)
        else None
    )
    return Param(p.name, p.kind, p.default, role, hint, marker)


class CallSpec:
    def __init__(self, function: Callable[..., Any]) -> None:
        if not callable(function):
            raise SpecError(f"thunk.fn expects a callable, got {function!r}")
        self.function = function
        self.name = getattr(function, "__qualname__", repr(function))
        try:
            self.signature = inspect.signature(function)
        except (ValueError, TypeError) as exc:
            raise SpecError(f"cannot inspect signature: {exc}") from exc
        ns = _namespace(_hint_target(function))
        self.params = tuple(
            _compile_param(p, ns, self.name) for p in self.signature.parameters.values()
        )
        self.by_name = {p.name: p for p in self.params}
        self.data = tuple(p for p in self.params if p.role == "data")
        self.static = tuple(p for p in self.params if p.role == "static")
        self.skip = tuple(p for p in self.params if p.role == "skip")
        self.output = _compile_param(
            inspect.Parameter(
                "output",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=self.signature.return_annotation,
            ),
            ns,
            self.name,
        )
        if self.output.role == "skip":
            raise SpecError("return Skip is not supported")

    def role(self, p: Param, value: Any) -> str:
        if p.role == "skip":
            return "skip"
        if p.marker is not None and p.marker.serializer is not None:
            return str(p.role)
        node = infer(value, p.name)
        if p.role == "static" and node.contains_array:
            raise ValueTypeError(
                f"{p.name}: Static requires finite JSON without arrays"
            )
        return p.role or ("data" if node.contains_array else "static")

    def bind(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs, opts = {}, {}
        for p in self.params:
            value = bound.arguments[p.name]
            role = self.role(p, value)
            if role != "skip":
                (inputs if role == "data" else opts)[p.name] = value
        return inputs, opts

    def check_group(self, values: Mapping[str, Any], role: str) -> None:
        for name in values:
            p = self.by_name.get(name)
            if p is None or p.role not in (None, role):
                raise TypeError(
                    f"{self.name}: unknown or role-mismatched {role} parameter {name!r}"
                )

    def check_pair(self, inputs: Mapping[str, Any], opts: Mapping[str, Any]) -> None:
        self.check_group(inputs, "data")
        self.check_group(opts, "static")
        if set(inputs) & set(opts):
            raise TypeError("duplicate names in inputs and opts")
        missing = [
            p.name
            for p in self.params
            if p.role != "skip" and p.name not in inputs and p.name not in opts
        ]
        if missing:
            raise TypeError(f"missing persisted parameters: {missing}")

    def merge(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        skipped: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        self.check_pair(inputs, opts)
        skipped = skipped or {}
        if set(skipped) - {p.name for p in self.skip}:
            raise TypeError("`skipped` only accepts Skip parameters")
        merged = dict(inputs) | dict(opts)
        for p in self.skip:
            if p.name in skipped:
                merged[p.name] = skipped[p.name]
            elif p.has_default:
                merged[p.name] = p.default
            else:
                raise TypeError(f"missing required skipped parameter {p.name!r}")
        return merged

    def to_call(
        self, values: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (
            tuple(
                values[p.name]
                for p in self.params
                if p.kind is inspect.Parameter.POSITIONAL_ONLY
            ),
            {
                p.name: values[p.name]
                for p in self.params
                if p.kind is not inspect.Parameter.POSITIONAL_ONLY
            },
        )


def _hint_target(function: Callable[..., Any]) -> Any:
    function = inspect.unwrap(function)
    while isinstance(function, functools.partial):
        function = inspect.unwrap(function.func)
    if inspect.isroutine(function) or isinstance(function, type):
        return function
    return type(function).__call__
