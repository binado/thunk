"""Runtime persistence and content-addressed caching."""

import json
import os
import stat
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import h5py

from . import _hdf5, _opts_json
from ._atomic import publish_all
from ._errors import (
    DigestMismatchError,
    SchemaMismatchError,
    SerializerContractError,
    StorageFormatError,
    ValueTypeError,
)
from ._fingerprint import (
    DIGEST_VERSION,
    _sha256,
    canonical_json,
    combined_digest,
    group_digest,
    json_digest,
)
from ._lock import Lock, LockedGroup, read_lock
from ._signature import CallSpec, Param
from ._spec import (
    Node,
    codec_identity,
    encode,
    infer,
    reconstruct,
    run_codec,
    validate_custom_data,
    validate_payload,
)

type PathLike = str | os.PathLike[str]
type Extras = Literal["forbid", "ignore"]
type Missing = Literal["raise", "default"]


@dataclass
class Prepared:
    role: str
    nodes: dict[str, Node]
    values: dict[str, Any]
    codecs: dict[str, Any]

    @property
    def representations(self) -> dict[str, Any]:
        return {
            k: {"node": n.describe(), "codec": self.codecs[k]}
            for k, n in self.nodes.items()
        }

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            k: _sha256(
                canonical_json({"name": k, "role": self.role, "representation": r})
            )
            for k, r in self.representations.items()
        }

    @property
    def digest(self) -> str:
        return (
            json_digest(self.values)
            if self.role == "static"
            else group_digest(self.nodes, self.values)
        )


def _prepare(
    params: tuple[Param, ...], values: Mapping[str, Any], role: str
) -> Prepared:
    nodes, payloads, codecs = {}, {}, {}
    for p in params:
        if p.name not in values:
            continue
        value = values[p.name]
        codec = codec_identity(p.marker)
        if codec is not None:
            value = run_codec(p.marker, value, p.name)
            if p.role == "data":
                validate_custom_data(value, p.name)
                # The Data contract accepts Mapping, storage uses ordinary dicts.
                value = _plain_mappings(value)
        try:
            node = infer(value, p.name)
            if (role == "static" or p.role == "static") and node.contains_array:
                raise ValueTypeError(
                    f"{p.name}: Static requires finite JSON without arrays"
                )
        except ValueTypeError as exc:
            if codec is not None:
                raise SerializerContractError(
                    f"{p.name}: invalid serializer payload: {exc}"
                ) from exc
            raise
        if role == "data" and p.role == "static" and codec is not None:
            value = encode(node, value, json_mode=True)
            node = infer(value, p.name)
        nodes[p.name] = node
        payloads[p.name] = encode(node, value, json_mode=role == "static")
        codecs[p.name] = codec
    return Prepared(role, nodes, payloads, codecs)


def _plain_mappings(value: Any) -> Any:
    return (
        {k: _plain_mappings(v) for k, v in value.items()}
        if isinstance(value, Mapping)
        else value
    )


def _digests(inputs: Prepared, opts: Prepared) -> dict[str, str | int]:
    a, b = inputs.digest, opts.digest
    fps = {"inputs": inputs.fingerprints, "opts": opts.fingerprints}
    return {
        "inputs_digest": a,
        "opts_digest": b,
        "digest": combined_digest(fps, a, b),
        "digest_version": DIGEST_VERSION,
    }


class FunctionPersistence[R]:
    """Inspect a callable once; infer storage independently for every operation.

    Ordinary annotations are optional loading hints and never affect cache keys.
    """

    def __init__(self, function: Callable[..., R], /) -> None:
        self._spec = CallSpec(function)

    def flatten(
        self, /, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply defaults and infer groups, dropping Skip; no codecs or array transfers."""
        return self._spec.bind(args, kwargs)

    def __call__(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        /,
        skipped: Mapping[str, Any] | None = None,
    ) -> R:
        """Assemble supplied values and execute, without reconstruction."""
        args, kwargs = self._spec.to_call(self._spec.merge(inputs, opts, skipped))
        return self._spec.function(*args, **kwargs)

    def _prepare_group(self, values: Mapping[str, Any], role: str) -> Prepared:
        self._spec.check_group(values, role)
        return _prepare(self._spec.params, values, role)

    @staticmethod
    def _writer(
        prepared: Prepared, kind: str, extra: Mapping[str, str | int] | None = None
    ) -> Callable[[Path], None]:
        def write(path: Path) -> None:
            with h5py.File(path, "w") as f:
                _hdf5.write_envelope(
                    f, kind, prepared.fingerprints, **dict(extra or {})
                )
                f.attrs["representations"] = json.dumps(prepared.representations)
                f.attrs["content_digest"] = prepared.digest
                group = (
                    f.create_group("inputs", track_order=True)
                    if kind == "inputs"
                    else f
                )
                for name, node in prepared.nodes.items():
                    _hdf5.write_node(group, name, node, prepared.values[name])

        return write

    def save_inputs(self, path: PathLike, inputs: Mapping[str, Any], /) -> None:
        """Atomically save supplied members in HDF5, enforcing explicit roles."""
        prepared = self._prepare_group(inputs, "data")
        publish_all([(path, self._writer(prepared, "inputs"))])

    def save_opts(self, path: PathLike, opts: Mapping[str, Any], /) -> None:
        """Atomically save supplied members as plain finite JSON."""
        prepared = self._prepare_group(opts, "static")
        publish_all([(path, lambda p: _opts_json.write_opts(p, prepared.values))])

    def save(
        self, input_file: PathLike, opts_file: PathLike, /, *args: Any, **kwargs: Any
    ) -> None:
        """Infer and prepare both groups before atomic publication of each file."""
        inputs, opts = self.flatten(*args, **kwargs)
        a, b = self._prepare_group(inputs, "data"), self._prepare_group(opts, "static")
        publish_all(
            [
                (input_file, self._writer(a, "inputs")),
                (opts_file, lambda p: _opts_json.write_opts(p, b.values)),
            ]
        )

    def _pair(
        self, inputs: Mapping[str, Any], opts: Mapping[str, Any]
    ) -> tuple[Prepared, Prepared]:
        self._spec.check_pair(inputs, opts)
        return self._prepare_group(inputs, "data"), self._prepare_group(opts, "static")

    def output_path(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        *,
        base_dir: PathLike | None = None,
    ) -> Path:
        """Hash concrete storage, codecs and content; exclude function identity and hints."""
        digests = _digests(*self._pair(inputs, opts))
        return Path(base_dir or ".") / f"{digests['digest']}.h5"

    def save_locked(
        self, input_file: PathLike, opts_file: PathLike, /, *args: Any, **kwargs: Any
    ) -> Path:
        """Prepare codecs once and publish data, JSON, then a version-2 lock."""
        inputs, opts = self.flatten(*args, **kwargs)
        a, b = self._pair(inputs, opts)
        digests = _digests(a, b)
        path = Path(opts_file).parent / f"{digests['digest']}.lock.json"

        def locked(file: PathLike, group: Prepared) -> LockedGroup:
            return LockedGroup(
                path=os.path.relpath(Path(file).absolute(), path.parent.absolute()),
                digest=group.digest,
                fingerprints=group.fingerprints,
                representations=group.representations,
            )

        lock = Lock(
            version=2,
            digest_version=2,
            digest=str(digests["digest"]),
            inputs=locked(input_file, a),
            opts=locked(opts_file, b),
        )
        publish_all(
            [
                (input_file, self._writer(a, "inputs")),
                (opts_file, lambda p: _opts_json.write_opts(p, b.values)),
                (path, lock.write),
            ]
        )
        return path

    @staticmethod
    def _stored(role: str, reps: Any, values: dict[str, Any], path: Path) -> Prepared:
        if not isinstance(reps, dict) or set(reps) != set(values):
            raise StorageFormatError(
                f"{path}: representations differ from stored names"
            )
        nodes, codecs = {}, {}
        for name, rep in reps.items():
            if not isinstance(rep, dict) or set(rep) != {"node", "codec"}:
                raise StorageFormatError(f"{path}: malformed representation for {name}")
            codec = rep["codec"]
            if codec is not None and (
                not isinstance(codec, dict)
                or set(codec) != {"kind", "serializer", "deserializer"}
                or not all(isinstance(v, str) for v in codec.values())
            ):
                raise StorageFormatError(f"{path}: malformed codec")
            node = Node.from_description(rep["node"])
            if role == "static" and node.contains_array:
                raise StorageFormatError(f"{path}: non-JSON representation")
            validate_payload(node, values[name], name, json_mode=role == "static")
            nodes[name], codecs[name] = node, codec
        return Prepared(role, nodes, {k: values[k] for k in nodes}, codecs)

    @classmethod
    def _read_hdf(cls, path: Path, kind: str) -> tuple[Prepared, dict[str, Any]]:
        f, fps = _hdf5.open_checked(path, kind)
        with f:
            try:
                reps = json.loads(_hdf5._attr(f, "representations", path))
                group = f.get("inputs") if kind == "inputs" else f
                if not isinstance(group, h5py.Group):
                    raise StorageFormatError(f"{path}: missing inputs group")
                decoded = {k: _hdf5.read_node(group[k], k) for k in group}
                prepared = cls._stored(
                    "data", reps, {k: v for k, (_, v) in decoded.items()}, path
                )
                if any(n != prepared.nodes[k] for k, (n, _) in decoded.items()):
                    raise StorageFormatError(
                        f"{path}: stored tags differ from representation"
                    )
                if fps != prepared.fingerprints:
                    raise SchemaMismatchError(
                        f"{path}: fingerprints differ from representations"
                    )
                if _hdf5._attr(f, "content_digest", path) != prepared.digest:
                    raise DigestMismatchError(f"{path}: content digest differs")
                return prepared, dict(f.attrs)
            except (KeyError, TypeError, ValueError) as exc:
                if isinstance(
                    exc,
                    (
                        StorageFormatError,
                        SchemaMismatchError,
                        DigestMismatchError,
                        ValueTypeError,
                    ),
                ):
                    raise
                raise StorageFormatError(
                    f"{path}: malformed stored structure: {exc}"
                ) from exc

    def _names(
        self,
        names: Mapping[str, Any],
        role: str,
        extras: Extras,
        missing: Missing,
        path: Path,
    ) -> dict[str, Any]:
        if extras not in ("forbid", "ignore") or missing not in ("raise", "default"):
            raise ValueError("invalid extras or missing policy")
        for name in names:
            p = self._spec.by_name.get(name)
            if p is None:
                if extras == "forbid":
                    raise SchemaMismatchError(f"{path}: unknown stored name {name!r}")
            elif p.role not in (None, role):
                raise SchemaMismatchError(f"{path}: {name!r} is now role {p.role!r}")
        filled = {}
        for p in self._spec.params:
            if p.role == role and p.name not in names:
                self._default(p, filled, missing, path)
        return filled

    @staticmethod
    def _default(
        p: Param, filled: dict[str, Any], missing: Missing, path: Path
    ) -> None:
        if missing != "default" or not p.has_default:
            raise SchemaMismatchError(f"{path}: missing stored value for {p.name!r}")
        filled[p.name] = p.default
        warnings.warn(
            f"{path}: {p.name!r} not stored; using current default {p.default!r}",
            stacklevel=4,
        )

    @staticmethod
    def _decode(
        p: Param,
        value: Any,
        *,
        hdf5: bool,
        codec: Any = None,
        check_codec: bool = False,
    ) -> Any:
        expected = codec_identity(p.marker)
        if check_codec and codec != expected:
            raise SchemaMismatchError(
                f"{p.name}: stored codec differs from current codec"
            )
        if expected is not None:
            return run_codec(p.marker, value, p.name, decode=True)
        return reconstruct(value, p.hint, p.name, hdf5=hdf5)

    def _restore(
        self, prepared: Prepared, path: Path, extras: Extras, missing: Missing
    ) -> dict[str, Any]:
        filled = self._names(prepared.values, prepared.role, extras, missing, path)
        return {
            p.name: self._decode(
                p,
                prepared.values[p.name],
                hdf5=prepared.role == "data",
                codec=prepared.codecs[p.name],
                check_codec=True,
            )
            if p.name in prepared.values
            else filled[p.name]
            for p in self._spec.params
            if p.name in prepared.values or p.name in filled
        }

    def load_inputs(
        self,
        path: PathLike,
        /,
        *,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> dict[str, Any]:
        """Restore HDF5; absence is checked only for explicitly Data parameters."""
        prepared, _ = self._read_hdf(Path(path), "inputs")
        return self._restore(prepared, Path(path), extras, missing)

    def load_opts(
        self,
        path: PathLike,
        /,
        *,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> dict[str, Any]:
        """Restore plain JSON; absence is checked only for explicitly Static parameters."""
        raw = _opts_json.read_opts(Path(path))
        filled = self._names(raw, "static", extras, missing, Path(path))
        return {
            p.name: self._decode(p, raw[p.name], hdf5=False)
            if p.name in raw
            else filled[p.name]
            for p in self._spec.params
            if p.name in raw or p.name in filled
        }

    def load(
        self,
        *,
        inputs: PathLike,
        opts: PathLike,
        skipped: Mapping[str, Any] | None = None,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Check completeness across files; default filling uses current defaults and warns."""
        a, b = (
            self.load_inputs(inputs, extras=extras, missing=missing),
            self.load_opts(opts, extras=extras, missing=missing),
        )
        if set(a) & set(b):
            raise SchemaMismatchError("duplicate names in inputs and opts")
        for p in self._spec.params:
            if p.role != "skip" and p.name not in a and p.name not in b:
                filled: dict[str, Any] = {}
                self._default(p, filled, missing, Path(opts))
                (a if self._spec.role(p, p.default) == "data" else b).update(filled)
        return self._spec.to_call(self._spec.merge(a, b, skipped))

    def load_lock(self, path: PathLike, /) -> tuple[dict[str, Any], dict[str, Any]]:
        """Verify representations and payloads before any user reconstruction code."""
        path = Path(path)
        lock = read_lock(path)
        a, _ = self._read_hdf(path.parent / lock.inputs.path, "inputs")
        raw = _opts_json.read_opts(path.parent / lock.opts.path)
        b = self._stored("static", lock.opts.representations, raw, path)
        for group, locked in ((a, lock.inputs), (b, lock.opts)):
            if (
                group.representations != locked.representations
                or group.fingerprints != locked.fingerprints
            ):
                raise SchemaMismatchError(f"{path}: locked representations differ")
            if group.digest != locked.digest:
                raise DigestMismatchError(f"{path}: content differs from lock")
            self._names(group.values, group.role, "forbid", "raise", path)
            for name in group.values:
                if group.codecs[name] != codec_identity(
                    self._spec.by_name[name].marker
                ):
                    raise SchemaMismatchError(f"{path}: {name} codec changed")
        try:
            self._spec.check_pair(a.values, b.values)
        except TypeError as exc:
            raise SchemaMismatchError(str(exc)) from exc
        if _digests(a, b)["digest"] != lock.digest:
            raise DigestMismatchError(f"{path}: combined digest differs from lock")
        return self._restore(a, path, "forbid", "raise"), self._restore(
            b, path, "forbid", "raise"
        )

    def save_output(
        self,
        path: PathLike,
        output: R,
        /,
        *,
        inputs: Mapping[str, Any] | None = None,
        opts: Mapping[str, Any] | None = None,
    ) -> None:
        """Store runtime output in HDF5, with optional input content metadata."""
        extra: dict[str, str | int] = {}
        if inputs is not None and opts is not None:
            extra = _digests(*self._pair(inputs, opts))
        elif inputs is not None:
            extra["inputs_digest"] = self._prepare_group(inputs, "data").digest
        elif opts is not None:
            extra["opts_digest"] = self._prepare_group(opts, "static").digest
        self._save_output(path, output, extra)

    def _save_output(
        self, path: PathLike, output: R, extra: Mapping[str, str | int]
    ) -> None:
        prepared = _prepare((self._spec.output,), {"output": output}, "data")
        publish_all([(path, self._writer(prepared, "output", extra))])

    def load_output(self, path: PathLike, /) -> R:
        return self._load_output(path)

    def _load_output(
        self, path: PathLike, expected: Mapping[str, str | int] | None = None
    ) -> R:
        prepared, attrs = self._read_hdf(Path(path), "output")
        if set(prepared.values) != {"output"}:
            raise StorageFormatError(f"{path}: missing output")
        for name, value in (expected or {}).items():
            if name not in attrs:
                raise StorageFormatError(f"{path}: missing {name}")
            stored = attrs[name]
            if name == "digest_version":
                if (
                    not isinstance(stored, Integral)
                    or isinstance(stored, bool)
                    or stored != DIGEST_VERSION
                ):
                    raise StorageFormatError(f"{path}: unsupported digest_version")
            elif (
                not isinstance(stored, str)
                or len(stored) != 64
                or any(c not in "0123456789abcdef" for c in stored)
            ):
                raise StorageFormatError(f"{path}: invalid {name}")
            if stored != value:
                raise DigestMismatchError(f"{path}: {name} differs from cache key")
        return self._decode(
            self._spec.output,
            prepared.values["output"],
            hdf5=True,
            codec=prepared.codecs["output"],
            check_codec=True,
        )

    def cached(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        /,
        *,
        outdir: PathLike,
        skipped: Mapping[str, Any] | None = None,
        refresh: bool = False,
    ) -> R:
        """Only absent entries miss. Hints may change restored values without changing keys."""
        merged = self._spec.merge(inputs, opts, skipped)
        digests = _digests(*self._pair(inputs, opts))
        directory = Path(outdir)
        path = directory / f"{digests['digest']}.h5"
        if not refresh:
            try:
                mode = path.stat().st_mode
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISDIR(mode):
                    raise IsADirectoryError(path)
                if not stat.S_ISREG(mode):
                    raise StorageFormatError(
                        f"{path}: cache entry is not a regular file"
                    )
                return self._load_output(path, digests)
        args, kwargs = self._spec.to_call(merged)
        result = self._spec.function(*args, **kwargs)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_output(path, result, digests)
        return result


def fn[R](function: Callable[..., R], /) -> FunctionPersistence[R]:
    """Inspect fixed signatures; ordinary annotations are optional reconstruction hints."""
    return FunctionPersistence(function)


def cache[**P, R](
    function: Callable[P, R], /, *, outdir: PathLike, refresh: bool = False
) -> Callable[P, R]:
    """Cache by runtime persistence. Use a directory specific to computation/revision."""
    persistence = fn(function)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        inputs, opts = persistence.flatten(*args, **kwargs)
        bound = persistence._spec.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return persistence.cached(
            inputs,
            opts,
            outdir=outdir,
            refresh=refresh,
            skipped={p.name: bound.arguments[p.name] for p in persistence._spec.skip},
        )

    return wrapped
