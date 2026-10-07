from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
from test_fresh_process import run_fresh

import thunk
from thunk import _atomic, _persistence


def test_interfaces_share_entries_and_refresh(tmp_path: Path) -> None:
    calls = []

    def f(
        x: np.ndarray, /, *, scale: int = 2, chunk: Annotated[int, thunk.Skip()] = 10
    ) -> tuple[np.ndarray, int]:
        calls.append(chunk)
        return x * scale, scale

    p = thunk.fn(f)
    x = np.arange(3)
    inputs, opts = p.flatten(x)
    wrapped = thunk.cache(f, outdir=tmp_path / "project/v1")
    assert wrapped.__wrapped__ is f  # ty: ignore[unresolved-attribute]
    assert not list(tmp_path.iterdir())
    np.testing.assert_array_equal(wrapped(x)[0], x * 2)
    result = p.cached(
        inputs, opts, outdir=tmp_path / "project/v1", skipped={"chunk": 20}
    )
    np.testing.assert_array_equal(result[0], x * 2)
    assert result[1] == 2
    assert calls == [10]
    forced = thunk.cache(f, outdir=tmp_path / "project/v1", refresh=True)
    forced(x, chunk=30)
    forced(x, chunk=40)
    assert calls == [10, 30, 40]
    wrapped(x, scale=3)
    x[0] = 99
    wrapped(x)
    assert len(calls) == 5


def test_output_directories_and_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)

    def f() -> None:
        calls.append(1)

    for outdir in ("relative/v1", tmp_path / "absolute/v1"):
        wrapped = thunk.cache(f, outdir=outdir)
        assert wrapped() is None
        assert thunk.fn(f).cached({}, {}, outdir=outdir) is None
        assert len(list(Path(outdir).glob("*.h5"))) == 1
    assert len(calls) == 2


def test_outdir_is_required() -> None:
    def f() -> int:
        raise AssertionError("executed")

    with pytest.raises(TypeError, match="outdir"):
        thunk.cache(f)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="outdir"):
        thunk.fn(f).cached({}, {})  # ty: ignore[missing-argument]


def test_function_arguments_do_not_collide(tmp_path: Path) -> None:
    def f(
        x: str,
        /,
        outdir: str,
        *,
        refresh: bool,
        skipped: Annotated[int, thunk.Skip()],
    ) -> str:
        assert skipped == 7
        return f"{x}/{outdir}/{refresh}"

    wrapped = thunk.cache(f, outdir=tmp_path / "f")
    assert wrapped("n", outdir="b", refresh=True, skipped=7) == "n/b/True"
    assert wrapped("n", outdir="b", refresh=True, skipped=8) == "n/b/True"
    with pytest.raises(TypeError):
        wrapped("n", outdir="b", refresh=True)  # ty: ignore[missing-argument]


def test_validation_on_hit_and_miss(tmp_path: Path) -> None:
    calls = []

    def f(x: int, skip: Annotated[int, thunk.Skip()]) -> int:
        calls.append(1)
        return x

    p = thunk.fn(f)
    for populated in (False, True):
        if populated:
            assert (
                p.cached({}, {"x": 1}, outdir=tmp_path / "f", skipped={"skip": 2}) == 1
            )
        for skipped in (None, {"wrong": 2}):
            with pytest.raises(TypeError):
                p.cached({}, {"x": 1}, outdir=tmp_path / "f", skipped=skipped)
        with pytest.raises(thunk.ValueTypeError):
            p.cached({}, {"x": 1.0}, outdir=tmp_path / "f", skipped={"skip": 2})
    assert calls == [1]

    def bad(x: int) -> object:
        raise AssertionError("executed")

    with pytest.raises(thunk.OutputCodecError):
        thunk.cache(bad, outdir=tmp_path / "f")
    with pytest.raises(thunk.OutputCodecError):
        thunk.fn(bad).cached({}, {"x": 1}, outdir=tmp_path / "f")


def test_cache_metadata_and_schema(tmp_path: Path) -> None:
    calls = []

    def f(x: int) -> int:
        calls.append(1)
        return x

    p = thunk.fn(f)
    path = p.output_path({}, {"x": 1}, base_dir=tmp_path / "f")
    wrapped = thunk.cache(f, outdir=tmp_path / "f")
    wrapped(1)
    original = path.read_bytes()
    for attr in ("digest", "inputs_digest", "opts_digest", "digest_version"):
        for value, error in [
            (None, thunk.StorageFormatError),
            ("bad", thunk.StorageFormatError),
            (
                2 if attr == "digest_version" else "0" * 64,
                thunk.StorageFormatError
                if attr == "digest_version"
                else thunk.DigestMismatchError,
            ),
        ]:
            path.write_bytes(original)
            with h5py.File(path, "r+") as h:
                del h.attrs[attr]
                if value is not None:
                    h.attrs[attr] = value
            with pytest.raises(error):
                wrapped(1)
    path.write_bytes(original)

    def other(x: int) -> str:
        raise AssertionError("executed")

    with pytest.raises(thunk.SchemaMismatchError):
        thunk.cache(other, outdir=tmp_path / "f")(1)
    path.write_text("not HDF5")
    with pytest.raises(thunk.StorageFormatError):
        wrapped(1)
    path.unlink()
    path.mkdir()
    with pytest.raises(IsADirectoryError):
        wrapped(1)
    assert calls == [1]


def test_failures_preserve_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = "ok"

    def f() -> int:
        if state == "raise":
            raise RuntimeError("failed")
        if state == "invalid":
            return "bad"  # ty: ignore[invalid-return-type]
        return 1

    p = thunk.fn(f)
    path = p.output_path({}, {}, base_dir=tmp_path / "f")
    wrapped = thunk.cache(f, outdir=tmp_path / "f", refresh=True)
    wrapped()
    original = path.read_bytes()
    for new_state, error in [
        ("raise", RuntimeError),
        ("invalid", thunk.ValueTypeError),
    ]:
        state = new_state
        with pytest.raises(error):
            wrapped()
        assert path.read_bytes() == original
        with pytest.raises(error):
            thunk.cache(f, outdir=tmp_path / "missing")()
        assert not list((tmp_path / "missing").iterdir())
    state = "ok"

    def fail_publish(temp: Path, dest: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(_atomic, "_publish", fail_publish)
    with pytest.raises(PermissionError):
        wrapped()
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_lookup_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def f() -> int:
        raise AssertionError("executed")

    def denied(self: Path, **kwargs: object) -> object:
        raise PermissionError("denied")

    wrapped = thunk.cache(f, outdir=tmp_path / "f")
    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(PermissionError):
        wrapped()


def test_digest_computed_once_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def f(x: np.ndarray) -> int:
        # Deliberately violate the no-mutation contract to expose rehashing.
        x[0] = 99
        return 1

    p = thunk.fn(f)
    x = np.array([1])
    inputs, opts = p.flatten(x)
    path = p.output_path(inputs, opts, base_dir=tmp_path / "f")
    original = p._digests
    calls = []

    def counted(inputs, opts):
        calls.append(1)
        return original(inputs, opts)

    monkeypatch.setattr(p, "_digests", counted)
    p.cached(inputs, opts, outdir=tmp_path / "f")
    assert calls == [1]
    with h5py.File(path) as h:
        assert h.attrs["digest"] == path.stem


def test_fresh_process_reuse(tmp_path: Path) -> None:
    def f(x: int) -> int:
        return x + 1

    thunk.cache(f, outdir=tmp_path / "f")(1)
    run_fresh(f"""
        import thunk
        def f(x: int) -> int:
            raise AssertionError("executed")
        assert thunk.cache(f, outdir={str(tmp_path / "f")!r})(1) == 2
    """)


def test_bound_callables_and_partials(tmp_path: Path) -> None:
    from functools import partial

    class Scale:
        def __init__(self, scale: int) -> None:
            self.scale = scale

        def __call__(self, x: int) -> int:
            return self.scale * x

        def method(self, x: int) -> int:
            return self.scale + x

    instance = Scale(3)
    cases: list[tuple[str, Callable[..., int], int]] = [
        ("instance", instance, 6),
        ("method", instance.method, 5),
        ("partial", partial(instance.method, x=2), 5),
    ]
    for name, target, expected in cases:
        wrapped = thunk.cache(target, outdir=tmp_path / name)
        assert wrapped(x=2) == expected
        assert wrapped(x=2) == expected


def test_serialization_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def f() -> int:
        return 1

    p = thunk.fn(f)
    path = p.output_path({}, {}, base_dir=tmp_path / "f")
    thunk.cache(f, outdir=tmp_path / "f")()
    original = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(_persistence._hdf5, "write_node", fail_write)
    for name in ("f", "missing"):
        with pytest.raises(RuntimeError, match="serialization failed"):
            thunk.cache(f, outdir=tmp_path / name, refresh=True)()
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    assert not list((tmp_path / "missing").iterdir())
