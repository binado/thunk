import json
from dataclasses import InitVar, dataclass, field

import numpy as np
import pytest

import thunk
from thunk._spec import Node, infer


@dataclass
class Inner:
    a: np.ndarray
    tags: list[str]


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


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        1,
        1.5,
        "x",
        [],
        (),
        {},
        [1, "a", None],
        {"a": (1, 2.0)},
        np.zeros(2),
        Outer(Inner(np.zeros(2), [])),
    ],
)
def test_runtime_descriptors(value):
    node = infer(value, "x")
    assert Node.from_description(json.loads(json.dumps(node.describe()))) == node


@pytest.mark.parametrize(
    "value",
    [object(), {1: 1}, b"x", {1}, np.array([object()]), WithInitVar(1, 2), NonInit()],
)
def test_unsupported_values(value):
    with pytest.raises(thunk.ValueTypeError, match="x"):
        infer(value, "x")


def test_nested_path_and_cycle():
    value = Outer(Inner(np.zeros(2), ["a", object()]))  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.ValueTypeError, match=r"x\.inner\.tags\[1\]"):
        infer(value, "x")
    cycle = []
    cycle.append({"again": cycle})
    with pytest.raises(thunk.ValueTypeError, match=r"x\[0\]\['again'\].*cycle"):
        infer(cycle, "x")


def test_contains_data():
    assert infer({"a": [float("inf")]}, "x").contains_array
    assert infer((1, np.zeros(0)), "x").contains_array
    assert not infer([{}, (), None], "x").contains_array
