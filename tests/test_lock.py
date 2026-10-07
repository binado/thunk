import json
import shutil
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
from pydantic import PlainSerializer, PlainValidator
from test_fresh_process import run_fresh

import thunk
from thunk import _atomic, _lock, _opts_json


def f(x: np.ndarray, seed: int = 42, *, lockfile: str = "yes") -> int:
    raise AssertionError("must not execute")


def test_roundtrip_and_output_agree(tmp_path: Path) -> None:
    p = thunk.fn(f)
    inputs, opts = p.flatten(np.arange(6).reshape(2, 3))
    expected = p.output_path(inputs, opts)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", **inputs, **opts)
    assert lock.name == expected.stem + ".lock.json"
    restored = p.load_lock(lock)
    assert p.output_path(*restored) == expected
    output = p.output_path(*restored, base_dir=tmp_path)
    p.save_output(output, 1, inputs=inputs, opts=opts)
    with h5py.File(output) as h:
        doc = json.loads(lock.read_text())
        assert h.attrs["digest"] == doc["digest"] == output.stem
        assert h.attrs["digest_version"] == 1
        assert h.attrs["inputs_digest"] == doc["inputs"]["digest"]
        assert h.attrs["opts_digest"] == doc["opts"]["digest"]
    p.save(tmp_path / "i.h5", tmp_path / "o.json", **inputs, **opts)
    assert p.load_opts(tmp_path / "o.json")["lockfile"] == "yes"


def test_path_is_pure_and_excludes_function_return_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other(
        x: np.ndarray,
        seed: int = 42,
        *,
        lockfile: str = "yes",
        skip: Annotated[int, thunk.Skip()] = 9,
    ) -> object:
        raise AssertionError

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("filesystem access")

    a, b = thunk.fn(f), thunk.fn(other)
    inputs, opts = a.flatten(np.zeros(1))
    monkeypatch.setattr(Path, "stat", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    key = a.output_path(inputs, opts)
    assert b.output_path(*b.flatten(np.zeros(1), skip=2)) == key
    assert (
        a.output_path(inputs, opts, base_dir=tmp_path / "missing")
        == tmp_path / "missing" / key
    )
    with pytest.raises(thunk.ValueTypeError):
        a.output_path(inputs, {**opts, "seed": 1.0})
    with pytest.raises(TypeError):
        a.output_path({}, opts)


def test_order_layout_dtype_shape_and_mutation() -> None:
    def g(x: np.ndarray, y: np.ndarray, a: int, b: dict[str, int]) -> None: ...

    p = thunk.fn(g)
    x = np.arange(6).reshape(2, 3)
    inputs, opts = p.flatten(x, np.zeros(1), 1, {"z": 1, "a": 2})
    key = p.output_path(inputs, opts)
    assert (
        p.output_path(
            dict(reversed(list(inputs.items()))), dict(reversed(list(opts.items())))
        )
        == key
    )
    for arr in (np.asfortranarray(x), x.astype(x.dtype.newbyteorder(">"))):
        assert p.output_path({**inputs, "x": arr}, opts) == key
    for arr in (x.astype(float), x.reshape(3, 2), x + 1):
        assert p.output_path({**inputs, "x": arr}, opts) != key
    assert p.output_path(inputs, {**opts, "b": {"a": 2, "z": 1}}) != key
    x[0, 0] = 99
    assert p.output_path(inputs, opts) != key


def test_empty_groups_and_schema_identity(tmp_path: Path) -> None:
    def empty() -> None: ...
    def a(n: int) -> None: ...
    def b(n: int | None) -> None: ...

    p = thunk.fn(empty)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json")
    assert p.load_lock(lock) == ({}, {})
    assert p.output_path({}, {}).stem == json.loads(lock.read_text())["digest"]
    assert thunk.fn(a).output_path({}, {"n": 1}) != thunk.fn(b).output_path(
        {}, {"n": 1}
    )


def test_fresh_process_key(tmp_path: Path) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.arange(3))
    actual = run_fresh(f"""
        import numpy as np
        import thunk
        from test_lock import f
        p = thunk.fn(f)
        print(p.output_path(*p.flatten(np.arange(3))))
        print(p.output_path(*p.load_lock({str(lock)!r})))
    """)
    assert actual.splitlines() == [p.output_path(*p.flatten(np.arange(3))).name] * 2


def test_relative_relocation_and_old_locks(tmp_path: Path) -> None:
    root = tmp_path / "old"
    (root / "opts").mkdir(parents=True)
    p = thunk.fn(f)
    lock = p.save_locked(root / "i.h5", root / "opts" / "o.json", np.zeros(1))
    assert json.loads(lock.read_text())["inputs"]["path"] == "../i.h5"
    shutil.move(root, tmp_path / "new")
    relocated = tmp_path / "new" / "opts" / lock.name
    assert p.load_lock(relocated)[1]["seed"] == 42
    p.save_locked(tmp_path / "new" / "i.h5", relocated.parent / "o.json", np.ones(1))
    assert relocated.exists()
    with pytest.raises(thunk.DigestMismatchError):
        p.load_lock(relocated)


@pytest.mark.parametrize(
    "target", ["inputs", "opts", "digest", "inputs_digest", "opts_digest"]
)
def test_tampering(tmp_path: Path, target: str) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    doc = json.loads(lock.read_text())
    if target == "inputs":
        with h5py.File(tmp_path / "i.h5", "r+") as h:
            h["inputs/x"][0] = 1
    elif target == "opts":
        (tmp_path / "o.json").write_text('{"seed": 43, "lockfile": "yes"}')
    else:
        if target == "digest":
            doc["digest"] = "0" * 64
        else:
            doc[target.removesuffix("_digest")]["digest"] = "0" * 64
        lock.write_text(json.dumps(doc))
    with pytest.raises(thunk.DigestMismatchError):
        p.load_lock(lock)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("version", 1.0),
        ("digest_version", "1"),
        ("digest_version", 2),
        ("digest", "A" * 64),
        ("digest", "a" * 63),
        ("unknown", 1),
        ("inputs", {}),
        ("opts", None),
    ],
)
def test_malformed_lock(tmp_path: Path, field: str, value: object) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    doc = json.loads(lock.read_text())
    doc[field] = value
    lock.write_text(json.dumps(doc))
    with pytest.raises(thunk.StorageFormatError):
        p.load_lock(lock)


@pytest.mark.parametrize(
    "field", ["version", "digest_version", "digest", "inputs", "opts"]
)
def test_required_fields(tmp_path: Path, field: str) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    doc = json.loads(lock.read_text())
    del doc[field]
    lock.write_text(json.dumps(doc))
    with pytest.raises(thunk.StorageFormatError):
        p.load_lock(lock)


def test_schema_and_hdf_fingerprints(tmp_path: Path) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))

    def changed(
        x: np.ndarray, seed: int | None = 42, *, lockfile: str = "yes"
    ) -> None: ...

    with pytest.raises(thunk.SchemaMismatchError):
        thunk.fn(changed).load_lock(lock)
    with h5py.File(tmp_path / "i.h5", "r+") as h:
        h.attrs["fingerprints"] = json.dumps({"x": "0" * 64})
    with pytest.raises(thunk.SchemaMismatchError):
        p.load_lock(lock)


@pytest.mark.parametrize("name", ["i.h5", "o.json", "lock"])
def test_missing_files(tmp_path: Path, name: str) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    (lock if name == "lock" else tmp_path / name).unlink()
    with pytest.raises(FileNotFoundError):
        p.load_lock(lock)


@pytest.mark.parametrize("writer", ["opts", "lock"])
def test_preparation_failure_preserves_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    before = {q: q.read_bytes() for q in tmp_path.iterdir()}

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        _opts_json if writer == "opts" else _lock.Lock,
        "write_opts" if writer == "opts" else "write",
        fail,
    )
    with pytest.raises(OSError):
        p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    assert {q: q.read_bytes() for q in tmp_path.iterdir()} == before
    assert lock.exists()


def test_publication_order_and_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = thunk.fn(f)
    seen = []
    publish = _atomic._publish

    def record(temp: Path, dest: Path) -> None:
        seen.append(dest)
        publish(temp, dest)

    monkeypatch.setattr(_atomic, "_publish", record)
    i, o = tmp_path / "i.h5", tmp_path / "o.json"
    lock = p.save_locked(i, o, np.zeros(1))
    assert seen == [i, o, lock]
    for a, b in [(i, i), (lock, o), (i, lock)]:
        with pytest.raises(ValueError, match="distinct"):
            p.save_locked(a, b, np.zeros(1))


def test_opts_verified_before_custom_validator_and_encoded_once(tmp_path: Path) -> None:
    calls = []

    def encode(v: int) -> int:
        calls.append(v)
        return v

    def decode(v: int) -> int:
        return v + 1

    def g(
        n: Annotated[
            int, thunk.Static(PlainSerializer(encode), PlainValidator(decode))
        ],
    ) -> None: ...

    p = thunk.fn(g)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", 1)
    assert calls == [1]
    assert p.load_lock(lock) == ({}, {"n": 2})
    assert calls == [1]


def test_custom_codecs_roundtrip(tmp_path: Path) -> None:
    from test_custom_serializers import Interval, Mesh, f

    p = thunk.fn(f)
    args = (Interval(0.5, 2.5), Mesh(np.arange(6.0).reshape(3, 2), "tri"))
    key = p.output_path(*p.flatten(*args))
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", *args)
    assert p.output_path(*p.load_lock(lock)) == key


def test_special_floats_and_top_level_json_order(tmp_path: Path) -> None:
    from test_opts_json import f, make_args

    p = thunk.fn(f)
    key = p.output_path(*p.flatten(**make_args()))
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", **make_args())
    o = tmp_path / "o.json"
    o.write_text(json.dumps(dict(reversed(list(json.loads(o.read_text()).items())))))
    assert p.output_path(*p.load_lock(lock)) == key


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_locked_options_require_exact_names(tmp_path: Path, change: str) -> None:
    p = thunk.fn(f)
    lock = p.save_locked(tmp_path / "i.h5", tmp_path / "o.json", np.zeros(1))
    o = tmp_path / "o.json"
    doc = json.loads(o.read_text())
    if change == "extra":
        doc["other"] = 1
    else:
        del doc["seed"]
    o.write_text(json.dumps(doc))
    with pytest.raises(thunk.SchemaMismatchError):
        p.load_lock(lock)


def test_parent_must_exist(tmp_path: Path) -> None:
    p = thunk.fn(f)
    with pytest.raises(FileNotFoundError):
        p.save_locked(tmp_path / "i.h5", tmp_path / "absent" / "o.json", np.zeros(1))
    assert list(tmp_path.iterdir()) == []


def test_signature_and_nested_input_order() -> None:
    def a(x: Annotated[dict[str, int], thunk.Data()], y: int, z: int) -> None: ...
    def b(x: Annotated[dict[str, int], thunk.Data()], z: int, y: int) -> None: ...

    inputs = {"x": {"a": 1, "b": 2}}
    opts = {"y": 1, "z": 2}
    key = thunk.fn(a).output_path(inputs, opts)
    assert thunk.fn(b).output_path(inputs, opts) != key
    assert thunk.fn(a).output_path({"x": {"b": 2, "a": 1}}, opts) != key
