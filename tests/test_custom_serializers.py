from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pytest
from pydantic import PlainSerializer, PlainValidator

import thunk
from thunk._fingerprint import param_fingerprint


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float


def _ser(v: Interval) -> list[float]:
    return [v.lo, v.hi]


def _val(raw: Any) -> Interval:
    lo, hi = raw
    return Interval(lo, hi)


IntervalStatic = Annotated[
    Interval,
    thunk.Static(PlainSerializer(_ser), PlainValidator(_val)),
]


class Mesh:
    def __init__(self, points: np.ndarray, name: str) -> None:
        self.points = points
        self.name = name


def _mesh_ser(m: Mesh) -> dict[str, Any]:
    return {"points": m.points, "meta": {"name": m.name, "n": len(m.points)}}


def _mesh_val(d: dict[str, Any]) -> Mesh:
    return Mesh(d["points"], d["meta"]["name"])


MeshData = Annotated[
    Mesh, thunk.Data(thunk.DataSerializer(_mesh_ser), thunk.DataValidator(_mesh_val))
]


def f(i: IntervalStatic, m: MeshData, k: int = 2) -> float:
    return i.hi - i.lo + len(m.points) * k


def test_pair_roundtrip(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    mesh = Mesh(np.arange(6.0).reshape(3, 2), "tri")
    args = (Interval(0.5, 2.5), mesh)
    pfn.save(tmp_path / "i.h5", tmp_path / "o.json", *args, k=3)
    a, kw = pfn.load(inputs=tmp_path / "i.h5", opts=tmp_path / "o.json")
    assert a == ()
    assert kw["i"] == Interval(0.5, 2.5) and type(kw["i"]) is Interval
    assert kw["m"].name == "tri"
    np.testing.assert_array_equal(kw["m"].points, mesh.points)
    assert f(*a, **kw) == f(*args, k=3)


def test_custom_static_goes_to_opts_and_data_to_inputs() -> None:
    spec = thunk.fn(f)._spec
    assert [p.name for p in spec.data] == ["m"]
    assert [p.name for p in spec.static] == ["i", "k"]


def test_fingerprint_includes_codec_names() -> None:
    def other_ser(v: Interval) -> list[float]:
        return [v.lo, v.hi]

    def g(
        i: Annotated[
            Interval, thunk.Static(PlainSerializer(other_ser), PlainValidator(_val))
        ],
    ) -> None: ...

    a = thunk.fn(f)._spec.by_name["i"]
    b = thunk.fn(g)._spec.by_name["i"]
    assert param_fingerprint(a) != param_fingerprint(b)


def test_static_marker_validation() -> None:
    ok_s, ok_v = PlainSerializer(_ser), PlainValidator(_val)
    with pytest.raises(thunk.SpecError, match="when_used"):
        thunk.Static(PlainSerializer(_ser, when_used="json"), ok_v)
    with pytest.raises(thunk.SpecError, match="PlainSerializer"):
        thunk.Static(_ser, ok_v)  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.SpecError, match="PlainValidator"):
        thunk.Static(ok_s, _val)  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.SpecError, match="together"):
        thunk.Static(ok_s)
    with pytest.raises(thunk.SpecError, match="together"):
        thunk.Static(validator=ok_v)


def test_data_marker_validation() -> None:
    with pytest.raises(thunk.SpecError, match="DataSerializer"):
        thunk.Data(_mesh_ser, thunk.DataValidator(_mesh_val))  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.SpecError, match="DataValidator"):
        thunk.Data(thunk.DataSerializer(_mesh_ser), _mesh_val)  # ty: ignore[invalid-argument-type]
    with pytest.raises(thunk.SpecError, match="together"):
        thunk.Data(thunk.DataSerializer(_mesh_ser))


def _data_pfn(ser: Any) -> Any:
    def g(
        m: Annotated[
            Mesh, thunk.Data(thunk.DataSerializer(ser), thunk.DataValidator(_mesh_val))
        ],
    ) -> None: ...

    return thunk.fn(g)


@pytest.mark.parametrize(
    "produced",
    [
        {"a/b": 1},
        {"": 1},
        {".": 1},
        {1: 1},
        {"a": np.array([object()], dtype=object)},
        {"a": [1, 2]},
        {"a": None},
        {"a": {"b": {"c/d": 1}}},
        [1, 2],
    ],
)
def test_bad_serializer_output_rejected_and_leaves_no_file(
    tmp_path: Path, produced: Any
) -> None:
    pfn = _data_pfn(lambda m: produced)
    with pytest.raises(thunk.SerializerContractError):
        pfn.save_inputs(tmp_path / "i.h5", {"m": Mesh(np.zeros(1), "x")})
    assert list(tmp_path.iterdir()) == []


def test_empty_nested_dict_and_scalar_types_roundtrip(tmp_path: Path) -> None:
    produced = {
        "e": {},
        "b": True,
        "i": 3,
        "f": float("inf"),
        "s": "x",
        "u": np.array(["ab"]),
    }
    seen: dict[str, Any] = {}

    def val(d: dict[str, Any]) -> Mesh:
        seen.update(d)
        return Mesh(np.zeros(1), "x")

    def g(
        m: Annotated[
            Mesh,
            thunk.Data(
                thunk.DataSerializer(lambda m: produced), thunk.DataValidator(val)
            ),
        ],
    ) -> None: ...

    pfn = thunk.fn(g)
    pfn.save_inputs(tmp_path / "i.h5", {"m": Mesh(np.zeros(1), "x")})
    pfn.load_inputs(tmp_path / "i.h5")
    assert list(seen) == ["e", "b", "i", "f", "s", "u"]
    assert seen["e"] == {} and seen["b"] is True and type(seen["i"]) is int
    assert seen["f"] == float("inf") and seen["u"].dtype.kind == "U"


def test_wrong_value_type_for_custom_rejected(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    with pytest.raises(thunk.ValueTypeError):
        pfn.save_inputs(tmp_path / "i.h5", {"m": "not a mesh"})
