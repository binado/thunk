import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import numpy as np
import pytest

import thunk
from thunk._hdf5 import escape_key, unescape_key


@dataclass(frozen=True)
class Pt:
    xy: np.ndarray
    label: str
    weight: float = 1.0


def roundtrip(tmp_path: Path, annotation: Any, value: Any) -> Any:
    def f(x): ...

    # force Data so scalar-only trees use HDF5
    pfn = thunk.fn(_with_role(f, annotation))
    path = tmp_path / "in.h5"
    pfn.save_inputs(path, {"x": value})
    return pfn.load_inputs(path)["x"]


def _with_role(f: Any, annotation: Any) -> Any:
    f.__annotations__ = {"x": Annotated[annotation, thunk.Data()]}
    return f


def same(a: Any, b: Any) -> None:
    assert type(a) is type(b)
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, Pt):
        same(a.xy, b.xy)
        same(a.label, b.label)
        same(a.weight, b.weight)
    else:
        assert a == b


@pytest.mark.parametrize(
    ("tp", "value"),
    [
        (list[int], []),
        (tuple[int, ...], ()),
        (tuple[()], ()),
        (dict[str, int], {}),
        (list[int], list(range(25))),
        (tuple[int, str, float], (1, "a", 2.5)),
        (tuple[str, ...], tuple("abcdefghijkl")),
        (int | None, None),
        (int | None, 3),
        (None, None),
        (list[int | None], [None, 1, None]),
        (bool, True),
        (int, 10**30),
        (str, "NaN"),
        (str, ""),
        (Literal["a", "b"], "b"),
        (Literal[1, 2], 2),
        (dict[str, int], {"": 1, "a/b": 2, ".": 3, "..": 4, "%": 5, "k": 6, "z\n": 7}),
        (dict[str, int], {"b": 1, "a": 2, "c": 3}),
        (dict[str, list[int]], {"x": [], "y": [1]}),
        (Pt, Pt(np.arange(4.0).reshape(2, 2), "p", 2.0)),
        (list[Pt], [Pt(np.zeros(1), "a"), Pt(np.ones(1), "b")]),
        (Pt | None, None),
    ],
)
def test_roundtrip(tmp_path: Path, tp: Any, value: Any) -> None:
    same(roundtrip(tmp_path, tp, value), value)


@pytest.mark.parametrize("x", [math.nan, math.inf, -math.inf, -0.0, 0.0, 1.5, 1e300])
def test_float_scalars(tmp_path: Path, x: float) -> None:
    out = roundtrip(tmp_path, float, x)
    assert type(out) is float
    assert math.copysign(1, out) == math.copysign(1, x) or math.isnan(x)
    same(out, x)


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(6).reshape(2, 3),
        np.array([1.5, np.nan, np.inf, -np.inf, -0.0]),
        np.array([True, False]),
        np.array([1 + 2j]),
        np.array(["ab", "cde", ""]),
        np.array(["ab", "日本", ""], dtype=">U2"),
        np.array([["é", "日本"], ["x", "yz"]]),
        np.array("scalar-str"),
        np.array(5.0),
        np.array([b"ab", b"c"]),
        np.zeros((0, 3)),
        np.zeros(0, dtype="U4"),
        np.arange(4, dtype=np.float16),
        np.arange(4, dtype=">i4"),
        np.arange(4, dtype=np.uint8),
    ],
)
def test_arrays(tmp_path: Path, arr: np.ndarray) -> None:
    out = roundtrip(tmp_path, np.ndarray, arr)
    assert out.dtype.kind == arr.dtype.kind
    assert out.shape == arr.shape
    assert out.dtype.itemsize == arr.dtype.itemsize
    np.testing.assert_array_equal(out, arr)


def test_noncontiguous_array(tmp_path: Path) -> None:
    arr = np.arange(12.0).reshape(3, 4)[:, ::2]
    np.testing.assert_array_equal(roundtrip(tmp_path, np.ndarray, arr), arr)


def test_object_array_rejected(tmp_path: Path) -> None:
    def f(x: np.ndarray) -> None: ...

    pfn = thunk.fn(f)
    with pytest.raises(thunk.ValueTypeError):
        pfn.save_inputs(tmp_path / "a.h5", {"x": np.array([1, "a"], dtype=object)})


def test_none_vs_optional_distinct_in_file(tmp_path: Path) -> None:
    def f(x: Any = None) -> None: ...

    def g(x: list[int] | None) -> None: ...

    pfn = thunk.fn(_with_role(g, list[int] | None))
    p1, p2 = tmp_path / "a.h5", tmp_path / "b.h5"
    pfn.save_inputs(p1, {"x": None})
    pfn.save_inputs(p2, {"x": []})
    with h5py.File(p1) as a, h5py.File(p2) as b:
        assert a["inputs/x"].attrs["kind"] == "none"
        assert b["inputs/x"].attrs["kind"] == "list"


def test_key_escaping_is_reversible() -> None:
    for key in ["", "a", "a/b", ".", "..", "%41", "%", "a%2Fb", "\x00", "é", "x y"]:
        esc = escape_key(key)
        assert "/" not in esc and esc and not esc.startswith(".")
        assert unescape_key(esc, "w") == key


def test_dict_order_preserved_in_file(tmp_path: Path) -> None:
    value = {"z": 1, "a": 2, "m": 3, "b": 4}
    assert list(roundtrip(tmp_path, dict[str, int], value)) == list(value)


def _inputs_pfn(tp: Any) -> Any:
    def f(x): ...

    return thunk.fn(_with_role(f, tp))


def _tamper(path: Path, fn: Any) -> None:
    with h5py.File(path, "r+") as f:
        fn(f)


@pytest.mark.parametrize(
    ("tp", "value", "tamper"),
    [
        (list[int], [1, 2, 3], lambda f: f["inputs/x"].__delitem__("1")),
        (list[int], [1, 2], lambda f: f["inputs/x"].move("1", "5")),
        (list[int], [1, 2], lambda f: f["inputs/x"].move("1", "01")),
        (list[int], [1, 2], lambda f: f["inputs/x"].move("1", "-1")),
        (list[int], [1], lambda f: f["inputs/x"].attrs.__setitem__("kind", "tuple")),
        (tuple[int, int], (1, 2), lambda f: f["inputs/x"].__delitem__("1")),
        (dict[str, int], {"a": 1}, lambda f: f["inputs/x"].move("ka", "a")),
        (dict[str, int], {"a": 1}, lambda f: f["inputs/x"].move("ka", "k%ZZ")),
        (Pt, Pt(np.zeros(1), "a"), lambda f: f["inputs/x"].__delitem__("label")),
        (int, 3, lambda f: f["inputs/x"].attrs.__setitem__("value", "3.0")),
        (int, 3, lambda f: f["inputs/x"].attrs.__setitem__("value", "oops")),
        (
            np.ndarray,
            np.zeros(2),
            lambda f: (f.__delitem__("inputs/x"), f["inputs"].create_group("x")),
        ),
        (int, 3, lambda f: f.attrs.__setitem__("storage_version", 99)),
        (int, 3, lambda f: f.attrs.__setitem__("thunk_format", "output")),
        (int, 3, lambda f: f.attrs.__delitem__("fingerprints")),
    ],
)
def test_malformed_files_rejected(
    tmp_path: Path, tp: Any, value: Any, tamper: Any
) -> None:
    pfn = _inputs_pfn(tp)
    path = tmp_path / "x.h5"
    pfn.save_inputs(path, {"x": value})
    _tamper(path, tamper)
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_inputs(path)


def test_untracked_dict_group_rejected(tmp_path: Path) -> None:
    pfn = _inputs_pfn(dict[str, int])
    path = tmp_path / "x.h5"
    pfn.save_inputs(path, {"x": {"a": 1}})

    def swap(f: h5py.File) -> None:
        g = f["inputs"].create_group("y")
        g.attrs["kind"] = "dict"
        f["inputs/x"].copy("ka", g)
        del f["inputs/x"]
        f["inputs"].move("y", "x")

    _tamper(path, swap)
    with pytest.raises(thunk.StorageFormatError, match="creation order"):
        pfn.load_inputs(path)


def test_not_hdf5_and_missing_file(tmp_path: Path) -> None:
    pfn = _inputs_pfn(int)
    bad = tmp_path / "bad.h5"
    bad.write_text("nope")
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_inputs(bad)
    with pytest.raises(FileNotFoundError):
        pfn.load_inputs(tmp_path / "missing.h5")
