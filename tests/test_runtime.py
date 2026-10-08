"""Regression coverage for operation-local inference and hint-only reconstruction."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import h5py
import numpy as np
import pytest
from pydantic import PlainSerializer, PlainValidator
from test_fresh_process import run_fresh

import thunk


@dataclass
class Box:
    value: tuple[int, str]
    extra: list[int] = field(default_factory=list)


def test_inspection_and_interleaved_operations(tmp_path):
    def f(x, y: Any = 2):
        raise AssertionError("executed")

    p = thunk.fn(f)
    a, b = p.flatten(1), p.flatten(np.zeros(2))
    assert a == ({}, {"x": 1, "y": 2})
    assert b[0]["x"].shape == (2,)
    p.save_inputs(tmp_path / "b.h5", b[0])
    p.save_opts(tmp_path / "a.json", a[1])
    assert p.load_opts(tmp_path / "a.json") == a[1]
    assert p.load_inputs(tmp_path / "b.h5")["x"].shape == (2,)

    def default(x=np.zeros(0)): ...

    assert default.__defaults__ is not None
    assert thunk.fn(default).flatten()[0]["x"] is default.__defaults__[0]

    def unsupported_default(x=object()): ...

    thunk.fn(unsupported_default)
    with pytest.raises(thunk.ValueTypeError):
        thunk.fn(unsupported_default).flatten()


def test_unresolved_hints_and_explicit_metadata():
    def f(x):
        raise AssertionError("executed")

    for hint in ("Unknown", Any, set[int], Literal["yes"]):
        f.__annotations__ = {"x": hint}
        assert thunk.fn(f).flatten(12) == ({}, {"x": 12})
    f.__annotations__ = {"x": "Annotated[Unknown, thunk.Data()]"}
    assert thunk.fn(f).flatten(12) == ({"x": 12}, {})
    f.__annotations__ = {"x": "Annotated[Unknown, thunk.Data(missing_codec)]"}
    with pytest.raises(thunk.SpecError):
        thunk.fn(f)
    f.__annotations__ = {"return": Annotated[Any, thunk.Skip()]}
    with pytest.raises(thunk.SpecError, match="return Skip"):
        thunk.fn(f)


def test_json_hints_and_fallback(tmp_path):
    def f(x): ...

    p = tmp_path / "opts.json"
    thunk.fn(f).save_opts(p, {"x": Box((1, "a"))})
    assert thunk.fn(f).load_opts(p) == {"x": {"value": [1, "a"], "extra": []}}
    for hint, expected in [
        (Box, Box((1, "a"))),
        (Box | None, Box((1, "a"))),
        (Box | int, {"value": [1, "a"], "extra": []}),
        (dict[str, tuple[Any, ...]], {"value": (1, "a"), "extra": ()}),
    ]:
        f.__annotations__ = {"x": hint}
        assert thunk.fn(f).load_opts(p)["x"] == expected
    p.write_text('{"x": {"value": [2, "b"]}}')
    f.__annotations__ = {"x": Box}
    assert thunk.fn(f).load_opts(p)["x"] == Box((2, "b"))
    for raw in ["{}", '{"unknown": 1}', "3"]:
        p.write_text('{"x": ' + raw + "}")
        assert not isinstance(thunk.fn(f).load_opts(p)["x"], Box)
    f.__annotations__ = {"x": tuple[int, int]}
    p.write_text('{"x": [1]}')
    assert thunk.fn(f).load_opts(p)["x"] == [1]
    p.write_text('{"x": [1, "no coercion"]}')
    assert thunk.fn(f).load_opts(p)["x"] == (1, "no coercion")


def test_nested_dataclasses_and_constructor_failure(tmp_path):
    def f(x: list[Box]): ...

    p = thunk.fn(f)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    p.save(i, o, [Box((1, "x"))])
    assert p.load_opts(o)["x"] == [Box((1, "x"))]

    @dataclass
    class Fails:
        value: Any

        def __post_init__(self):
            raise RuntimeError("constructor")

    def failing(x: Fails): ...

    o.write_text('{"x": {"value": 1}}')
    with pytest.raises(thunk.ReconstructionError, match="x:.*constructor"):
        thunk.fn(failing).load_opts(o)


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), {"a": [1, float("inf")]}]
)
def test_nonfinite_and_explicit_roles(tmp_path, value):
    def f(x): ...

    p = thunk.fn(f)
    assert list(p.flatten(value)[0]) == ["x"]
    p.save(tmp_path / "i.h5", tmp_path / "o.json", value)
    assert (tmp_path / "o.json").read_text().strip() == "{}"
    assert "x" in p.load_inputs(tmp_path / "i.h5")
    f.__annotations__ = {"x": Annotated[Any, thunk.Static()]}
    with pytest.raises(thunk.ValueTypeError, match="Static"):
        thunk.fn(f).flatten(value)
    f.__annotations__ = {
        "x": Annotated[
            Any,
            thunk.Static(
                PlainSerializer(lambda v: "finite"), PlainValidator(lambda v: value)
            ),
        ]
    }
    thunk.fn(f).save_opts(tmp_path / "o.json", {"x": value})
    assert thunk.fn(f).load_opts(tmp_path / "o.json")["x"] is value


def test_empty_heterogeneous_and_structural_identity(tmp_path):
    def f(x):
        return x

    p = thunk.fn(f)
    values = [
        [],
        (),
        {},
        [1, "a", None, {"b": []}],
        Box((1, "a")),
        {"value": (1, "a"), "extra": []},
    ]
    keys = [p.output_path(*p.flatten(v)) for v in values]
    assert len(set(keys)) == len(values)
    for index, value in enumerate(values):
        file = tmp_path / f"{index}.h5"
        p.save_inputs(file, {"x": value})
        loaded = p.load_inputs(file)["x"]
        if not isinstance(value, Box):
            assert loaded == value and type(loaded) is type(value)
        else:
            assert loaded == {"value": (1, "a"), "extra": []}


def test_custom_precedence_once_and_context(tmp_path):
    calls = []

    def serialize(v):
        calls.append("serialize")
        return {"value": v}

    def deserialize(v):
        calls.append("deserialize")
        return ("custom", v["value"])

    def f(
        x: Annotated[
            Box,
            thunk.Data(
                thunk.DataSerializer(serialize), thunk.DataDeserializer(deserialize)
            ),
        ],
    ):
        return x

    p = thunk.fn(f)
    p.flatten(1)
    assert calls == []
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", 1)
    assert calls == ["serialize"]
    assert p.load_lock(lock)[0] == {"x": ("custom", 1)}
    assert calls == ["serialize", "deserialize"]
    calls.clear()
    p.cached({"x": 1}, {}, outdir=tmp_path / "cache")
    assert calls == ["serialize"]
    assert thunk.DataDeserializer is thunk.DataValidator

    def bad(v):
        raise RuntimeError("bad codec")

    f.__annotations__ = {
        "x": Annotated[
            Any, thunk.Data(thunk.DataSerializer(bad), thunk.DataValidator(deserialize))
        ]
    }
    with pytest.raises(
        thunk.SerializerContractError, match="x: serializer failed.*bad codec"
    ):
        thunk.fn(f).save_inputs(tmp_path / "bad.h5", {"x": 1})


def test_partial_groups_and_combined_completeness(tmp_path):
    def f(x, y=2): ...

    p = thunk.fn(f)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    p.save_inputs(i, {"x": 1})
    p.save_opts(o, {})
    assert p.load_opts(o, missing="default") == {}
    with pytest.raises(thunk.SchemaMismatchError, match="y"):
        p.load(inputs=i, opts=o)
    with pytest.warns(UserWarning, match="y"):
        assert p.load(inputs=i, opts=o, missing="default") == ((), {"x": 1, "y": 2})
    p.save_opts(o, {"x": 3})
    with pytest.raises(thunk.SchemaMismatchError, match="duplicate"):
        p.load(inputs=i, opts=o)
    with pytest.raises(TypeError, match="duplicate"):
        p.output_path({"x": 1}, {"x": 1, "y": 2})
    with pytest.raises(TypeError, match="missing"):
        p.output_path({"x": 1}, {})


def test_cache_hints_do_not_change_key_or_require_return_annotations(tmp_path):
    calls = []

    def f(x):
        calls.append(1)
        return Box((x, "a"))

    old = thunk.fn(f)
    key = old.output_path(*old.flatten(1))
    old.cached(*old.flatten(1), outdir=tmp_path)
    f.__annotations__ = {"x": float, "return": Box}
    new = thunk.fn(f)
    assert new.output_path(*new.flatten(1)) == key
    assert new.cached(*new.flatten(1), outdir=tmp_path) == Box((1, "a"))
    assert old.cached(*old.flatten(1), outdir=tmp_path) == {
        "value": (1, "a"),
        "extra": [],
    }
    assert calls == [1]


def test_output_codecs_and_compatibility(tmp_path):
    def f() -> Annotated[
        Any,
        thunk.Static(
            PlainSerializer(lambda v: [v]), PlainValidator(lambda v: ("restored", v[0]))
        ),
    ]:
        return 3

    p = thunk.fn(f)
    assert p.cached({}, {}, outdir=tmp_path) == 3
    assert p.cached({}, {}, outdir=tmp_path) == ("restored", 3)

    def other():
        raise AssertionError("executed")

    with pytest.raises(thunk.SchemaMismatchError, match="codec"):
        thunk.cache(other, outdir=tmp_path)()


def test_versions_and_reconstruction_after_lock_verification(tmp_path):
    calls = []

    @dataclass
    class Tracked:
        value: int

        def __post_init__(self):
            calls.append(1)

    def f(x: Tracked): ...

    p = thunk.fn(f)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    lock = p.save_locked(i, o, Tracked(1))
    calls.clear()
    o.write_text('{"x": {"value": 2}}')
    with pytest.raises(thunk.DigestMismatchError):
        p.load_lock(lock)
    assert calls == []
    with h5py.File(i, "r+") as file:
        file.attrs["storage_version"] = 1
    with pytest.raises(thunk.StorageFormatError, match="storage_version"):
        p.load_inputs(i)


def test_unannotated_fresh_process(tmp_path):
    def f(x):
        return x

    p = thunk.fn(f)
    p.save(tmp_path / "i.h5", tmp_path / "o.json", {"a": [np.arange(2), (), 1]})
    run_fresh(f"""
        import thunk
        def f(x): raise AssertionError('executed')
        p = thunk.fn(f)
        args, kw = p.load(inputs={str(tmp_path / "i.h5")!r}, opts={str(tmp_path / "o.json")!r})
        assert kw['x']['a'][0].tolist() == [0, 1]
        assert kw['x']['a'][1] == ()
    """)


def test_empty_tuple_hint_and_ordinary_forward_reference(tmp_path):
    def f(x): ...

    for hint in ("DatabaseConnection", "StaticConfiguration", "SkipReason"):
        f.__annotations__ = {"x": hint}
        assert thunk.fn(f).flatten(1) == ({}, {"x": 1})
    path = tmp_path / "opts.json"
    f.__annotations__ = {"x": tuple[()]}
    path.write_text('{"x": [1]}')
    assert thunk.fn(f).load_opts(path)["x"] == [1]
    path.write_text('{"x": []}')
    assert thunk.fn(f).load_opts(path)["x"] == ()


def test_alias_hints_and_data_output_codec(tmp_path):
    type Pair = tuple[int, str]

    def f(x: Pair): ...

    path = tmp_path / "opts.json"
    path.write_text('{"x": [1, "a"]}')
    assert thunk.fn(f).load_opts(path)["x"] == (1, "a")
    calls = []

    def ser(v):
        calls.append("encode")
        return {"v": v}

    def dec(v):
        calls.append("decode")
        return v["v"] + 1

    def g() -> Annotated[
        Any, thunk.Data(thunk.DataSerializer(ser), thunk.DataDeserializer(dec))
    ]:
        return 2

    wrapped = thunk.cache(g, outdir=tmp_path / "cache")
    assert wrapped() == 2
    assert wrapped() == 3
    assert calls == ["encode", "decode"]
    g.__annotations__ = {
        "return": Annotated[
            Any, thunk.Static(PlainSerializer(ser), PlainValidator(dec))
        ]
    }
    with pytest.raises(thunk.SchemaMismatchError, match="codec"):
        thunk.cache(g, outdir=tmp_path / "cache")()


def test_signed_nan_and_hdf5_hint_precedence(tmp_path):
    def f(x: tuple[int, ...]): ...

    p = thunk.fn(f)
    path = tmp_path / "i.h5"
    for value in ([1, 2], float("-nan")):
        p.save_inputs(path, {"x": value})
        loaded = p.load_inputs(path)["x"]
        assert type(loaded) is type(value)

    def opposite(x: list[int]): ...

    thunk.fn(opposite).save_inputs(path, {"x": (1, 2)})
    assert thunk.fn(opposite).load_inputs(path)["x"] == (1, 2)


def test_unresolved_nested_and_aliased_metadata():
    def f(x): ...

    f.__annotations__ = {
        "x": "Annotated[list[Annotated[Unknown, thunk.Data()]], thunk.Skip()]"
    }
    with pytest.raises(thunk.SpecError, match="top-level"):
        thunk.fn(f)
    f.__globals__["DiskRole"] = thunk.Data
    try:
        f.__annotations__ = {"x": "Annotated[Unknown, DiskRole(missing_codec)]"}
        with pytest.raises(thunk.SpecError, match="metadata"):
            thunk.fn(f)
    finally:
        del f.__globals__["DiskRole"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_plain_json_must_decode_to_finite_values(tmp_path, literal):
    def f(x): ...

    path = tmp_path / "opts.json"
    path.write_text('{"x": ' + literal + "}")
    with pytest.raises(thunk.StorageFormatError):
        thunk.fn(f).load_opts(path)
