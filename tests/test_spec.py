import json
from dataclasses import InitVar, dataclass, field
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pytest

import thunk
from thunk._spec import (
    Array,
    DataclassNode,
    DictNode,
    FixedTupleNode,
    ListNode,
    LiteralNode,
    NoneNode,
    OptionalNode,
    Scalar,
    VarTupleNode,
    compile_annotation,
)

type Vec = list[float]
type Rec = list[Rec]


@dataclass
class Inner:
    a: np.ndarray
    tags: list[str]
    ClassMarker: ClassVar[int] = 3


@dataclass
class Outer:
    inner: Inner
    opt: Inner | None = None


@dataclass
class WithInitVar:
    x: int
    y: InitVar[int]


@dataclass
class NonInit:
    x: int = field(init=False, default=1)


@dataclass
class SelfRef:
    child: "SelfRef | None" = None


@pytest.mark.parametrize(
    ("tp", "cls"),
    [
        (int, Scalar),
        (None, NoneNode),
        (np.ndarray, Array),
        (npt.NDArray, Array),
        (npt.NDArray[np.float64], Array),
        (Literal["a", "b"], LiteralNode),
        (int | None, OptionalNode),
        (list[int], ListNode),
        (tuple[int, ...], VarTupleNode),
        (tuple[int, str], FixedTupleNode),
        (tuple[()], FixedTupleNode),
        (dict[str, float], DictNode),
        (Inner, DataclassNode),
        (Vec, ListNode),
        (Annotated[int, "doc"], Scalar),
    ],
)
def test_supported(tp: Any, cls: type) -> None:
    assert isinstance(compile_annotation(tp), cls)


@pytest.mark.parametrize(
    "tp",
    [
        Any,
        list,
        dict,
        tuple,
        dict[int, int],
        int | str,
        int | str | None,
        set[int],
        bytes,
        npt.NDArray[np.float64, np.int64],  # ty: ignore[invalid-type-arguments]
        Literal[1.5],  # ty: ignore[invalid-type-form]
        WithInitVar,
        NonInit,
        SelfRef,
        Rec,
        list[Annotated[int, thunk.Skip()]],
        Inner | Annotated[int, thunk.Data()] | None,
    ],
)
def test_unsupported(tp: Any) -> None:
    with pytest.raises(thunk.SpecError):
        compile_annotation(tp)


def test_contains_array_propagates() -> None:
    assert compile_annotation(Outer).contains_array
    assert not compile_annotation(dict[str, list[int]]).contains_array
    assert compile_annotation(tuple[int, np.ndarray]).contains_array


def test_strict_validation_messages() -> None:
    node = compile_annotation(Outer)
    good = Outer(Inner(np.zeros(2), ["a"]))
    node.validate(good, "x")
    bad = Outer(Inner(np.zeros(2), ["a", 3]))  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.ValueTypeError, match=r"x\.inner\.tags\[1\]"):
        node.validate(bad, "x")


@pytest.mark.parametrize(
    ("tp", "value"),
    [
        (float, 1),
        (int, True),
        (int, 1.0),
        (list[int], (1,)),
        (tuple[int, ...], [1]),
        (tuple[int, int], (1,)),
        (dict[str, int], {1: 1}),
        (np.ndarray, [1, 2]),
        (np.ndarray, np.array([object()], dtype=object)),
        (Literal["a"], "b"),
        (Literal[1], True),
    ],
)
def test_strict_rejections(tp: Any, value: Any) -> None:
    with pytest.raises(thunk.ValueTypeError):
        compile_annotation(tp).validate(value, "v")


def test_describe_is_json_and_stable() -> None:
    d = compile_annotation(Outer).describe()
    assert json.dumps(d, sort_keys=True) == json.dumps(
        compile_annotation(Outer).describe(), sort_keys=True
    )
    assert d["kind"] == "dataclass"
