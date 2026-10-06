import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import pytest

import thunk


@dataclass
class Grid:
    shape: tuple[int, int]
    steps: list[float]
    mode: Literal["a", "b"] = "a"


def f(
    grid: Grid,
    window: tuple[float, ...],
    limit: float,
    count: int,
    flag: bool,
    name: str,
    maybe: int | None,
    table: dict[str, float],
    fixed: tuple[int, str] = (1, "x"),
) -> None: ...


def make_args() -> dict:
    return dict(
        grid=Grid((2, 3), [0.5, math.inf]),
        window=(1.0, -math.inf, math.nan),
        limit=-0.0,
        count=7,
        flag=True,
        name="NaN",
        maybe=None,
        table={"z": 1.0, "a": math.inf, "m": 2.0},
    )


def test_roundtrip_restores_types(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    inputs, opts = pfn.flatten(**make_args())
    assert inputs == {}
    path = tmp_path / "o.json"
    pfn.save_opts(path, opts)
    out = pfn.load_opts(path)
    assert out["grid"] == opts["grid"] and type(out["grid"]) is Grid
    assert type(out["grid"].shape) is tuple
    assert out["window"][:2] == (1.0, -math.inf) and math.isnan(out["window"][2])
    assert type(out["window"]) is tuple
    assert out["limit"] == 0.0 and math.copysign(1, out["limit"]) == -1
    assert out["count"] == 7 and type(out["count"]) is int
    assert out["flag"] is True
    assert out["name"] == "NaN"
    assert out["maybe"] is None
    assert list(out["table"]) == ["z", "a", "m"]
    assert out["fixed"] == (1, "x")
    assert list(out) == list(opts)


def test_file_is_strict_json_with_envelope(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    _, opts = pfn.flatten(**make_args())
    path = tmp_path / "o.json"
    pfn.save_opts(path, opts)
    doc = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(c))
    assert doc["thunk_format"] == "opts" and doc["storage_version"] == 1
    assert set(doc["fingerprints"]) == set(doc["values"])
    assert doc["values"]["window"] == [1.0, "-Infinity", "NaN"]
    assert list(doc["values"]["table"]) == ["z", "a", "m"]


def test_int_float_strictness_on_write(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    _, opts = pfn.flatten(**make_args())
    for name, bad in [("limit", 1), ("count", 1.0), ("count", True), ("flag", 1)]:
        with pytest.raises(thunk.ValueTypeError):
            pfn.save_opts(tmp_path / "x.json", {**opts, name: bad})
    assert not (tmp_path / "x.json").exists()


def _edit(path: Path, name: str, raw: object) -> None:
    doc = json.loads(path.read_text())
    doc["values"][name] = raw
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("count", 1.5),
        ("count", 1.0),
        ("count", True),
        ("count", "7"),
        ("flag", 1),
        ("limit", "oops"),
        ("window", "nope"),
        ("grid", {"shape": [1], "steps": [], "mode": "a"}),
        ("grid", {"shape": [1, 2], "steps": [], "mode": "zzz"}),
        ("table", {"a": "x"}),
    ],
)
def test_invalid_stored_values_rejected(tmp_path: Path, name: str, raw: object) -> None:
    pfn = thunk.fn(f)
    _, opts = pfn.flatten(**make_args())
    path = tmp_path / "o.json"
    pfn.save_opts(path, opts)
    _edit(path, name, raw)
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_opts(path)


def test_integral_float_literal_for_float_param_loads_as_float(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    _, opts = pfn.flatten(**make_args())
    path = tmp_path / "o.json"
    pfn.save_opts(path, opts)
    _edit(path, "limit", 1)
    assert type(pfn.load_opts(path)["limit"]) is float


@pytest.mark.parametrize("text", ["{", "[]", '{"thunk_format": "inputs"}'])
def test_bad_envelope(tmp_path: Path, text: str) -> None:
    path = tmp_path / "o.json"
    path.write_text(text)
    with pytest.raises(thunk.StorageFormatError):
        thunk.fn(f).load_opts(path)


def test_bad_version(tmp_path: Path) -> None:
    pfn = thunk.fn(f)
    _, opts = pfn.flatten(**make_args())
    path = tmp_path / "o.json"
    pfn.save_opts(path, opts)
    doc = json.loads(path.read_text())
    doc["storage_version"] = 2
    path.write_text(json.dumps(doc))
    with pytest.raises(thunk.StorageFormatError, match="storage_version"):
        pfn.load_opts(path)


def test_opts_digest_distinguishes_int_float_and_order(tmp_path: Path) -> None:
    def g(a: float, b: dict[str, int]) -> int:
        return 0

    pfn = thunk.fn(g)
    digests = []
    for a, b in [
        (1.0, {"x": 1, "y": 2}),
        (-0.0, {"x": 1, "y": 2}),
        (0.0, {"x": 1, "y": 2}),
        (1.0, {"y": 2, "x": 1}),
    ]:
        _, opts = pfn.flatten(a, b)
        p = tmp_path / f"o{len(digests)}.h5"
        pfn.save_output(p, 0, opts=opts)
        with h5py.File(p) as h:
            digests.append(h.attrs["opts_digest"])
    assert len(set(digests)) == 4
