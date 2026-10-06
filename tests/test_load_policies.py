import math
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

import thunk


def v1(x: np.ndarray, n: int = 1) -> int:
    return n


def v2(x: np.ndarray, n: int = 1, extra: int = 9, *, req: float) -> int:
    return n + extra


def v2_default_only(x: np.ndarray, n: int = 1, extra: int = 9) -> int:
    return n + extra


def v_removed(x: np.ndarray) -> int:
    return 0


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    thunk.fn(v1).save(i, o, np.arange(3.0), n=4)
    return i, o


def test_extras_forbid_and_ignore(files: tuple[Path, Path]) -> None:
    i, o = files
    pfn = thunk.fn(v_removed)
    pfn.load_inputs(i)
    with pytest.raises(thunk.SchemaMismatchError, match="'n'"):
        pfn.load_opts(o)
    assert pfn.load_opts(o, extras="ignore") == {}


def test_missing_raise_default_and_warn(files: tuple[Path, Path]) -> None:
    i, o = files
    pfn = thunk.fn(v2_default_only)
    with pytest.raises(thunk.SchemaMismatchError, match="extra"):
        pfn.load_opts(o)
    with pytest.warns(UserWarning, match="extra"):
        opts = pfn.load_opts(o, missing="default")
    assert opts == {"n": 4, "extra": 9}
    with pytest.warns(UserWarning):
        args, kwargs = pfn.load(inputs=i, opts=o, missing="default")
    assert kwargs["extra"] == 9 and kwargs["n"] == 4


def test_missing_without_default_still_raises(files: tuple[Path, Path]) -> None:
    _, o = files
    with pytest.raises(thunk.SchemaMismatchError, match="req"):
        thunk.fn(v2).load_opts(o, missing="default")


def test_bad_policy_values(files: tuple[Path, Path]) -> None:
    i, _ = files
    with pytest.raises(ValueError):
        thunk.fn(v1).load_inputs(i, extras="nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        thunk.fn(v1).load_inputs(i, missing="nope")  # ty: ignore[invalid-argument-type]


def test_role_mismatch_always_raises(files: tuple[Path, Path]) -> None:
    i, o = files

    def moved(
        x: Annotated[np.ndarray, thunk.Skip()], n: Annotated[int, thunk.Data()] = 1
    ) -> int:
        return n

    pfn = thunk.fn(moved)
    # x stored as data but now Skip; n stored in opts but now Data
    with pytest.raises(thunk.SchemaMismatchError, match="role"):
        pfn.load_inputs(i, extras="ignore")
    with pytest.raises(thunk.SchemaMismatchError, match="role"):
        pfn.load_opts(o, extras="ignore")
    with pytest.raises(thunk.SchemaMismatchError, match="role"):
        pfn.load_opts(o, extras="ignore", missing="default")


def test_changed_annotation_is_schema_mismatch(files: tuple[Path, Path]) -> None:
    i, o = files

    def changed(x: Annotated[list[float], thunk.Data()], n: float = 1.0) -> int:
        return 0

    pfn = thunk.fn(changed)
    with pytest.raises(thunk.SchemaMismatchError, match="'n'"):
        pfn.load_opts(o)

    def changed_x(x: Annotated[np.ndarray, thunk.Data()], n: str = "") -> int:
        return 0

    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(changed_x).load_opts(o)

    def array_to_list(x: list[int]) -> int:
        return 0

    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(array_to_list).load_inputs(i, extras="ignore")


def test_fingerprint_compared_only_over_stored_names(files: tuple[Path, Path]) -> None:
    _, o = files
    # `extra` added later with a different annotation never mattered to the file
    with pytest.warns(UserWarning):
        thunk.fn(v2_default_only).load_opts(o, missing="default")


def test_wrong_file_kind(files: tuple[Path, Path], tmp_path: Path) -> None:
    i, o = files
    pfn = thunk.fn(v1)
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_inputs(o)
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_opts(i)
    with pytest.raises(thunk.StorageFormatError):
        pfn.load_output(i)


def test_skipped_in_load(files: tuple[Path, Path]) -> None:
    i, o = files

    def g(x: np.ndarray, n: int = 1, *, s: Annotated[int, thunk.Skip()] = 5) -> int:
        return n + s

    pfn = thunk.fn(g)
    # stored under v1 which had no `s`: fine, s is skipped
    args, kwargs = pfn.load(inputs=i, opts=o, skipped={"s": 10})
    assert kwargs["s"] == 10 and g(*args, **kwargs) == 14
    with pytest.raises(TypeError):
        pfn.load(inputs=i, opts=o, skipped={"n": 1})


def test_stored_skip_name_raises(tmp_path: Path) -> None:
    def a(x: int, s: int = 1) -> None: ...
    def b(x: int, s: Annotated[int, thunk.Skip()] = 1) -> None: ...

    p = tmp_path / "o.json"
    thunk.fn(a).save_opts(p, thunk.fn(a).flatten(1)[1])
    with pytest.raises(thunk.SchemaMismatchError, match="role"):
        thunk.fn(b).load_opts(p, extras="ignore")


def test_input_missing_default_policy(tmp_path: Path) -> None:
    def old(x: np.ndarray) -> None: ...

    def new(x: np.ndarray, y: Annotated[int, thunk.Data()] = 3) -> None: ...

    p = tmp_path / "i.h5"
    thunk.fn(old).save_inputs(p, {"x": np.zeros(2)})
    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(new).load_inputs(p)
    with pytest.warns(UserWarning):
        out = thunk.fn(new).load_inputs(p, missing="default")
    assert out["y"] == 3 and list(out) == ["x", "y"]


def test_output_roundtrip_and_digests(tmp_path: Path) -> None:
    pfn = thunk.fn(v1)
    inputs, opts = pfn.flatten(np.arange(3.0), n=2)
    p = tmp_path / "out.h5"
    pfn.save_output(p, 5, inputs=inputs, opts=opts)
    assert pfn.load_output(p) == 5
    with h5py.File(p) as f:
        d1, d2 = f.attrs["inputs_digest"], f.attrs["opts_digest"]
    pfn.save_output(p, 5, inputs={"x": np.arange(3.0)}, opts={"n": 2})
    with h5py.File(p) as f:
        assert (f.attrs["inputs_digest"], f.attrs["opts_digest"]) == (d1, d2)
    pfn.save_output(p, 5, inputs={"x": np.arange(3.0) + 1}, opts={"n": 2})
    with h5py.File(p) as f:
        assert f.attrs["inputs_digest"] != d1
    pfn.save_output(p, 5)
    with h5py.File(p) as f:
        assert "inputs_digest" not in f.attrs


def test_output_codec_errors(tmp_path: Path) -> None:
    def no_ann(x: int): ...  # noqa: ANN201

    def bad(x: int) -> object: ...

    def none(x: int) -> None: ...

    for func in (no_ann, bad):
        pfn = thunk.fn(func)  # input persistence must still work
        pfn.save_opts(tmp_path / "o.json", {"x": 1})
        with pytest.raises(thunk.OutputCodecError):
            pfn.save_output(tmp_path / "x.h5", None)
        with pytest.raises(thunk.OutputCodecError):
            pfn.load_output(tmp_path / "x.h5")
    pfn = thunk.fn(none)
    pfn.save_output(tmp_path / "n.h5", None)
    assert pfn.load_output(tmp_path / "n.h5") is None


def test_output_structured_and_schema_check(tmp_path: Path) -> None:
    def a(x: int) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def b(x: int) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    p = tmp_path / "o.h5"
    thunk.fn(a).save_output(p, (np.arange(2), math.nan))
    arr, nan = thunk.fn(a).load_output(p)
    assert math.isnan(nan) and arr.tolist() == [0, 1]
    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(b).load_output(p)
    with pytest.raises(thunk.ValueTypeError):
        thunk.fn(a).save_output(p, [np.arange(2), 1.0])  # ty: ignore[invalid-argument-type]


def test_save_input_validation(tmp_path: Path) -> None:
    pfn = thunk.fn(v1)
    with pytest.raises(thunk.ValueTypeError):
        pfn.save_inputs(tmp_path / "i.h5", {"x": [1, 2]})
    with pytest.raises(TypeError, match="unknown"):
        pfn.save_inputs(tmp_path / "i.h5", {"x": np.zeros(1), "z": 1})
    with pytest.raises(TypeError, match="missing"):
        pfn.save_opts(tmp_path / "o.json", {})
