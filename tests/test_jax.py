from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import h5py
import numpy as np
import pytest
from test_fresh_process import run_fresh

import thunk
from thunk import _jax
from thunk._fingerprint import _Hasher, fingerprint, group_digest
from thunk._spec import Array, JaxArray, compile_annotation

if TYPE_CHECKING:
    import jax
else:
    jax = pytest.importorskip("jax")


def never_run(x: jax.Array, n: int = 1) -> jax.Array:
    raise AssertionError("function body executed")


@dataclass
class Mixed:
    numpy: np.ndarray
    arrays: dict[str, list[jax.Array | None]]
    keys: tuple[jax.Array, ...]


def nested(x: tuple[Mixed, jax.Array | None]) -> tuple[Mixed, jax.Array | None]:
    raise AssertionError("function body executed")


def assert_array(actual: Any, expected: Any) -> None:
    assert isinstance(actual, jax.Array)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
        assert jax.random.key_impl(actual) == jax.random.key_impl(expected)
        np.testing.assert_array_equal(
            jax.random.key_data(actual), jax.random.key_data(expected)
        )
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", sorted(_jax.NUMERIC_DTYPES, key=str))
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
def test_numeric_roundtrip(
    tmp_path: Path, dtype: np.dtype, shape: tuple[int, ...]
) -> None:
    with jax.enable_x64():
        raw = np.arange(np.prod(shape), dtype=np.int64).reshape(shape).astype(dtype)
        if dtype.kind == "c":
            raw += (raw * 2j).astype(dtype)
        x = jax.device_put(raw)
        p = thunk.fn(never_run)
        p.save(tmp_path / "i.h5", tmp_path / "o.json", x)
        args, kwargs = p.load(inputs=tmp_path / "i.h5", opts=tmp_path / "o.json")
        assert args == () and kwargs["n"] == 1
        assert_array(kwargs["x"], x)
        p.save_output(tmp_path / "out.h5", x, inputs={"x": x})
        assert_array(p.load_output(tmp_path / "out.h5"), x)
        assert not kwargs["x"].weak_type


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", "complex64", "complex128"]
)
def test_nonfinite(tmp_path: Path, dtype: str) -> None:
    with jax.enable_x64():
        x = jax.numpy.asarray([np.nan, np.inf, -np.inf, -0.0], dtype=dtype)
        p = thunk.fn(never_run)
        p.save_output(tmp_path / "o.h5", x)
        assert_array(p.load_output(tmp_path / "o.h5"), x)


@pytest.mark.parametrize("implementation", list(_jax.KEY_SHAPES))
@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (0,), (2, 0)])
def test_key_roundtrip(
    tmp_path: Path, implementation: str, shape: tuple[int, ...]
) -> None:
    key = jax.random.key(17, impl=implementation)
    x = jax.random.split(key, shape) if shape else key
    p = thunk.fn(never_run)
    path = tmp_path / "keys.h5"
    p.save_inputs(path, {"x": x})
    default = "rbg" if implementation != "rbg" else "threefry2x32"
    with jax.default_prng_impl(default):
        restored = p.load_inputs(path)["x"]
    assert_array(restored, x)
    for original, loaded in zip(x.reshape(-1), restored.reshape(-1), strict=True):
        np.testing.assert_array_equal(
            jax.random.normal(original, (5,)), jax.random.normal(loaded, (5,))
        )
    p.save_output(tmp_path / "out.h5", x, inputs={"x": x})
    assert_array(p.load_output(tmp_path / "out.h5"), x)
    assert group_digest(p._spec.data, {"x": x}) == group_digest(
        p._spec.data, {"x": restored}
    )
    with h5py.File(path) as f:
        ds = f["inputs/x"]
        assert ds.attrs["jax_kind"] == "prng_key"
        assert ds.attrs["jax_impl"] == implementation
        assert ds.dtype == np.dtype("uint32")


def test_nested_and_digests(tmp_path: Path) -> None:
    value = (
        Mixed(np.arange(3), {"a/b": [jax.numpy.ones(2), None]}, (jax.random.key(3),)),
        None,
    )
    p = thunk.fn(nested)
    inputs, opts = p.flatten(value)
    assert inputs["x"] is value and opts == {}
    p.save_inputs(tmp_path / "i.h5", inputs)
    loaded = p.load_inputs(tmp_path / "i.h5")
    actual = loaded["x"]
    np.testing.assert_array_equal(actual[0].numpy, value[0].numpy)
    assert_array(actual[0].arrays["a/b"][0], value[0].arrays["a/b"][0])
    assert actual[0].arrays["a/b"][1] is actual[1] is None
    assert_array(actual[0].keys[0], value[0].keys[0])
    for name, source in [("before", inputs), ("after", loaded)]:
        p.save_output(tmp_path / f"{name}.h5", value, inputs=source)
    with h5py.File(tmp_path / "before.h5") as a, h5py.File(tmp_path / "after.h5") as b:
        assert a.attrs["inputs_digest"] == b.attrs["inputs_digest"]
    assert_array(p.load_output(tmp_path / "before.h5")[0].keys[0], value[0].keys[0])


def test_annotations_and_schema() -> None:
    assert compile_annotation(jax.Array) == JaxArray()
    assert JaxArray().contains_array
    assert JaxArray().describe() == {"kind": "jax_array"}
    assert Array().describe() == {"kind": "array"}
    assert fingerprint("x", "data", JaxArray()) != fingerprint("x", "data", Array())
    for annotation in (jax.typing.ArrayLike, jax.Array | np.ndarray):
        with pytest.raises(thunk.SpecError):
            compile_annotation(annotation)

    def static(x: Annotated[jax.Array, thunk.Static()]) -> None: ...

    with pytest.raises(thunk.SpecError, match="Static but contains arrays"):
        thunk.fn(static)


def test_jit_and_flatten_do_not_execute_or_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = jax.numpy.ones(2)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("host transfer")

    for function in (
        never_run,
        jax.jit(never_run),
        functools.partial(jax.jit(never_run), n=2),
    ):
        p = thunk.fn(function)
        with monkeypatch.context() as m:
            m.setattr(jax, "device_get", forbidden)
            inputs, _ = p.flatten(x)
            assert inputs["x"] is x
            JaxArray().validate(x, "x")
        p.save_inputs(tmp_path / "i.h5", inputs)
        assert_array(p.load_inputs(tmp_path / "i.h5")["x"], x)


@pytest.mark.parametrize(
    "kind",
    [
        "numpy",
        "weak",
        "deleted",
        "bfloat16",
        "float8_e4m3fn",
        "tracer",
        "nonaddressable",
    ],
)
def test_invalid_arrays_preserve_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    p = thunk.fn(never_run)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    p.save(i, o, jax.numpy.ones(2))
    before = i.read_bytes(), o.read_bytes()
    if kind == "numpy":
        x, message = np.ones(2), "expected jax.Array"
    elif kind == "weak":
        x, message = jax.numpy.asarray(1), "weakly typed.*explicit dtype"
    elif kind == "deleted":
        x = jax.numpy.ones(2)
        x.delete()
        message = "deleted.*live array"
    elif kind == "tracer":

        def tracing(x: jax.Array) -> jax.Array:
            p.save(i, o, x)
            return x

        with pytest.raises(thunk.ValueTypeError, match="x:.*tracers.*outside"):
            jax.jit(tracing)(jax.numpy.ones(2))
        assert (i.read_bytes(), o.read_bytes()) == before
        return
    elif kind == "nonaddressable":
        x, message = jax.numpy.ones(2), "not fully addressable.*gather"
        monkeypatch.setattr(
            type(x), "is_fully_addressable", property(lambda self: False)
        )
    else:
        x, message = jax.numpy.ones(2, dtype=kind), "unsupported JAX dtype.*cast"
    with pytest.raises(thunk.ValueTypeError, match=f"x:.*{message}"):
        p.save(i, o, x)
    assert (i.read_bytes(), o.read_bytes()) == before
    assert sorted(f.name for f in tmp_path.iterdir()) == ["i.h5", "o.json"]


def test_wrong_numpy_type_and_nested_path(tmp_path: Path) -> None:
    def numpy_only(x: np.ndarray) -> None: ...

    with pytest.raises(thunk.ValueTypeError, match="x: expected numpy.ndarray"):
        thunk.fn(numpy_only).save_inputs(tmp_path / "i.h5", {"x": jax.numpy.ones(2)})
    value = (Mixed(np.ones(2), {"bad": [jax.numpy.asarray(1)]}, ()), None)
    with pytest.raises(
        thunk.ValueTypeError, match=r"x\[0\].arrays\['bad'\]\[0\]:.*weakly"
    ):
        thunk.fn(nested).save_inputs(tmp_path / "i.h5", {"x": value})


def test_legacy_key_and_digest_tags(tmp_path: Path) -> None:
    x = jax.random.PRNGKey(0)
    p = thunk.fn(never_run)
    p.save_inputs(tmp_path / "i.h5", {"x": x})
    assert_array(p.load_inputs(tmp_path / "i.h5")["x"], x)
    with h5py.File(tmp_path / "i.h5") as f:
        assert f["inputs/x"].attrs["jax_kind"] == "array"
    values = [
        (Array(), np.asarray(x)),
        (JaxArray(), x),
        (JaxArray(), jax.random.key(0)),
    ]
    digests = []
    for node, value in values:
        h = _Hasher()
        h.feed(node, value)
        digests.append(h.hexdigest())
    assert len(set(digests)) == 3
    # Same bits and shape, different RNG algorithm.
    assert group_digest(
        p._spec.data, {"x": jax.random.key(0, impl="rbg")}
    ) != group_digest(p._spec.data, {"x": jax.random.key(0, impl="unsafe_rbg")})


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_kind",
        "bad_kind",
        "array_kind",
        "missing_impl",
        "unknown_impl",
        "array_impl",
        "dtype",
        "shape",
        "scalar",
        "group",
        "null",
        "numeric_string",
    ],
)
def test_malformed_storage(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "i.h5"
    p = thunk.fn(never_run)
    p.save_inputs(path, {"x": jax.random.key(1)})
    with h5py.File(path, "r+") as f:
        ds = f["inputs/x"]
        if corruption == "missing_kind":
            del ds.attrs["jax_kind"]
        elif corruption in ("bad_kind", "array_kind"):
            ds.attrs["jax_kind"] = (
                "bogus" if corruption == "bad_kind" else ["array", "prng_key"]
            )
        elif corruption == "missing_impl":
            del ds.attrs["jax_impl"]
        elif corruption in ("unknown_impl", "array_impl"):
            ds.attrs["jax_impl"] = (
                "custom" if corruption == "unknown_impl" else ["rbg", "unsafe_rbg"]
            )
        else:
            del f["inputs/x"]
            if corruption == "group":
                f.create_group("inputs/x")
            else:
                if corruption == "null":
                    ds = f.create_dataset("inputs/x", dtype="uint32")
                else:
                    raw = {
                        "dtype": np.ones(2, dtype="float32"),
                        "shape": np.ones(3, dtype="uint32"),
                        "scalar": np.array(1, dtype="uint32"),
                        "numeric_string": np.array([b"a"]),
                    }[corruption]
                    ds = f.create_dataset("inputs/x", data=raw)
                ds.attrs["jax_kind"] = (
                    "array" if corruption == "numeric_string" else "prng_key"
                )
                ds.attrs["jax_impl"] = "threefry2x32"
    with pytest.raises(thunk.StorageFormatError, match="x:"):
        p.load_inputs(path)


def test_schema_mismatch(tmp_path: Path) -> None:
    def numpy_only(x: np.ndarray) -> np.ndarray:
        raise AssertionError

    path = tmp_path / "i.h5"
    thunk.fn(numpy_only).save_inputs(path, {"x": np.ones(2)})
    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(never_run).load_inputs(path)
    thunk.fn(never_run).save_inputs(path, {"x": jax.numpy.ones(2)})
    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(numpy_only).load_inputs(path)


def test_x64_fresh_processes(tmp_path: Path) -> None:
    run_fresh(f"""
        import jax
        import numpy as np
        import thunk
        jax.config.update("jax_enable_x64", True)
        def f(x: jax.Array) -> jax.Array:
            raise AssertionError("body executed")
        p = thunk.fn(f)
        for dtype in ["int64", "uint64", "float64", "complex128"]:
            x = jax.numpy.asarray([2**40], dtype=dtype)
            p.save_inputs({str(tmp_path)!r} + "/" + dtype + ".h5", {{"x": x}})
            y = p.load_inputs({str(tmp_path)!r} + "/" + dtype + ".h5")["x"]
            assert y.dtype == x.dtype
            np.testing.assert_array_equal(x, y)
    """)
    run_fresh(f"""
        import jax
        import thunk
        jax.config.update("jax_enable_x64", False)
        def f(x: jax.Array): ...
        for dtype in ["int64", "uint64", "float64", "complex128"]:
            try:
                thunk.fn(f).load_inputs({str(tmp_path)!r} + "/" + dtype + ".h5")
            except thunk.ValueTypeError as exc:
                assert "JAX_ENABLE_X64" in str(exc)
            else:
                raise AssertionError("silently narrowed " + dtype)
    """)


def test_importing_jax_after_thunk_compilation() -> None:
    run_fresh("""
        import sys
        import numpy as np
        import thunk
        def numpy_only(x: np.ndarray): ...
        thunk.fn(numpy_only)
        assert "jax" not in sys.modules
        import jax
        def jax_only(x: jax.Array): ...
        p = thunk.fn(jax_only)
        x = jax.numpy.ones(2)
        assert p.flatten(x)[0]["x"] is x
    """)


@pytest.mark.parametrize("name", ["custom_test", "threefry2x32"])
def test_custom_prng_rejected(tmp_path: Path, name: str) -> None:
    from jax.extend.random import define_prng_impl

    def unused(*args: Any) -> Any:
        raise AssertionError("custom RNG executed")

    implementation = define_prng_impl(
        key_shape=(2,),
        seed=unused,
        split=unused,
        random_bits=unused,
        fold_in=unused,
        name=name,
        tag="test",
    )
    key = jax.random.wrap_key_data(
        jax.numpy.zeros(2, dtype="uint32"), impl=implementation
    )
    with pytest.raises(
        thunk.ValueTypeError, match="x: unsupported PRNG implementation"
    ):
        thunk.fn(never_run).save_inputs(tmp_path / "i.h5", {"x": key})


def test_failed_host_transfer_preserves_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = thunk.fn(never_run)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    x = jax.numpy.ones(2)
    p.save(i, o, x)
    before = i.read_bytes(), o.read_bytes()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(jax, "device_get", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        p.save(i, o, x)
    assert (i.read_bytes(), o.read_bytes()) == before
    assert sorted(f.name for f in tmp_path.iterdir()) == ["i.h5", "o.json"]


def test_locked_key_and_output_path(tmp_path: Path) -> None:
    def f(x: jax.Array, key: jax.Array, n: int = 1) -> jax.Array:
        raise AssertionError("must not run")

    p = thunk.fn(f)
    args = (jax.numpy.arange(4, dtype=jax.numpy.float32), jax.random.key(42))
    key = p.output_path(*p.flatten(*args))
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", *args)
    assert p.output_path(*p.load_lock(lock)) == key


def test_cached_jax_output(tmp_path: Path) -> None:
    calls = []

    def f(x: jax.Array) -> jax.Array:
        calls.append(1)
        return x + 1

    cached = thunk.cache(f, outdir=tmp_path / "jax/v1")
    x = jax.numpy.arange(3, dtype=jax.numpy.int32)
    assert_array(cached(x), x + 1)
    assert_array(cached(x), x + 1)
    assert calls == [1]
