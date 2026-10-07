"""``FunctionPersistence`` and the ``fn`` entry point."""

import os
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import h5py

from . import _hdf5, _opts_json
from ._atomic import publish_all
from ._errors import DigestMismatchError, SchemaMismatchError, StorageFormatError
from ._fingerprint import (
    DIGEST_VERSION,
    combined_digest,
    group_digest,
    json_digest,
    output_fingerprint,
    param_fingerprint,
)
from ._lock import Lock, LockedGroup, read_lock
from ._signature import CallSpec, Param
from ._spec import Node


def _node(p: Param) -> Node:
    assert p.node is not None
    return p.node


type PathLike = str | os.PathLike[str]
type Extras = Literal["forbid", "ignore"]
type Missing = Literal["raise", "default"]


class FunctionPersistence[R]:
    """Generated save/load methods for one annotated callable.

    Build instances with :func:`thunk.fn`.
    """

    def __init__(self, function: Callable[..., R], /) -> None:
        self._spec = CallSpec(function)
        self._opts = _opts_json.OptsCodec(self._spec.static)

    # -- flatten / call --------------------------------------------------

    def flatten(
        self, /, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Bind arguments and split them into ``(inputs, opts)``.

        Follows ordinary Python binding and fills defaults. Never calls the
        wrapped callable and never copies arrays; ``Skip`` arguments are dropped.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            ``Data`` parameters and ``Static`` parameters, each in signature order.

        Raises
        ------
        TypeError
            If the arguments do not bind to the signature.
        """
        return self._spec.bind(args, kwargs)

    def __call__(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        /,
        skipped: Mapping[str, Any] | None = None,
    ) -> R:
        """Merge the groups with execution-only values and call the callable.

        Parameters
        ----------
        inputs, opts
            Groups as returned by ``flatten`` or the ``load_*`` methods.
        skipped
            Values for ``Skip`` parameters; omitted ones use current defaults.

        Raises
        ------
        TypeError
            On unknown or missing names, non-``Skip`` names in ``skipped``, or a
            required ``Skip`` parameter that was not supplied.
        """
        merged = self._spec.merge(inputs, opts, skipped)
        args, kwargs = self._spec.to_call(merged)
        return self._spec.function(*args, **kwargs)

    # -- encoding helpers ------------------------------------------------

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        self._spec.check_names(inputs, self._spec.data, "inputs")
        for p in self._spec.data:
            assert p.node is not None
            p.node.validate(inputs[p.name], p.name)

    def _inputs_writer(self, inputs: Mapping[str, Any]) -> Callable[[Path], None]:
        self._validate_inputs(inputs)
        fps = {p.name: param_fingerprint(p) for p in self._spec.data}

        def write(path: Path) -> None:
            with h5py.File(path, "w") as f:
                _hdf5.write_envelope(f, "inputs", fps)
                group = f.create_group("inputs")
                for p in self._spec.data:
                    assert p.node is not None
                    _hdf5.write_node(group, p.name, p.node, inputs[p.name])

        return write

    def _opts_writer(self, opts: Mapping[str, Any]) -> Callable[[Path], None]:
        self._spec.check_names(opts, self._spec.static, "opts")
        encoded = self._opts.encode(opts)
        return lambda path: _opts_json.write_opts(path, encoded)

    # -- save ------------------------------------------------------------

    def save_inputs(self, path: PathLike, inputs: Mapping[str, Any], /) -> None:
        """Validate ``inputs`` and write them atomically to an HDF5 file.

        Raises
        ------
        ValueTypeError
            If a value does not match its annotation.
        TypeError
            If ``inputs`` has unknown or missing names.
        """
        publish_all([(path, self._inputs_writer(inputs))])

    def save_opts(self, path: PathLike, opts: Mapping[str, Any], /) -> None:
        """Validate ``opts`` and write them atomically to a JSON file.

        Raises
        ------
        ValueTypeError
            If a value does not match its annotation.
        TypeError
            If ``opts`` has unknown or missing names.
        """
        publish_all([(path, self._opts_writer(opts))])

    def save(
        self, input_file: PathLike, opts_file: PathLike, /, *args: Any, **kwargs: Any
    ) -> None:
        """Flatten the arguments, then save both groups.

        Both files are encoded before either destination is replaced, so an
        encoding failure leaves both untouched. The two replacements are not a
        single transaction. Never calls the wrapped callable.

        Raises
        ------
        ValueError
            If the two destinations are the same file.
        """
        inputs, opts = self.flatten(*args, **kwargs)
        publish_all(
            [
                (input_file, self._inputs_writer(inputs)),
                (opts_file, self._opts_writer(opts)),
            ]
        )

    def _fingerprints(self) -> dict[str, dict[str, str]]:
        return {
            "inputs": {p.name: param_fingerprint(p) for p in self._spec.data},
            "opts": {p.name: param_fingerprint(p) for p in self._spec.static},
        }

    def _digests(
        self, inputs: Mapping[str, Any] | None, opts: Mapping[str, Any] | None
    ) -> tuple[dict[str, str | int], dict[str, Any] | None]:
        extra: dict[str, str | int] = {}
        encoded = None
        if inputs is not None:
            self._validate_inputs(inputs)
            extra["inputs_digest"] = group_digest(self._spec.data, inputs)
        if opts is not None:
            self._spec.check_names(opts, self._spec.static, "opts")
            encoded = self._opts.encode(opts)
            extra["opts_digest"] = json_digest(encoded)
        if inputs is not None and opts is not None:
            extra["digest"] = combined_digest(
                self._fingerprints(),
                str(extra["inputs_digest"]),
                str(extra["opts_digest"]),
            )
            extra["digest_version"] = DIGEST_VERSION
        return extra, encoded

    def output_path(
        self,
        inputs: Mapping[str, Any],
        opts: Mapping[str, Any],
        *,
        base_dir: PathLike | None = None,
    ) -> Path:
        """Return a content/schema-derived filename without touching the filesystem.

        Function identity, return annotations, and skipped arguments are excluded.
        Use a computation-specific base directory to distinguish implementations.
        """
        digests, _ = self._digests(inputs, opts)
        return (
            Path(base_dir) / f"{digests['digest']}.h5"
            if base_dir is not None
            else Path(f"{digests['digest']}.h5")
        )

    def save_locked(
        self, input_file: PathLike, opts_file: PathLike, /, *args: Any, **kwargs: Any
    ) -> Path:
        """Save both groups and publish their lockfile last; return its path.

        All three files are prepared before any replacement. Replacements are
        individually atomic, not a transaction. Parent directories must exist.
        Custom serializers must produce deterministic representations.
        """
        inputs, opts = self.flatten(*args, **kwargs)
        digests, encoded = self._digests(inputs, opts)
        assert encoded is not None
        path = Path(opts_file).parent / f"{digests['digest']}.lock.json"
        fps = self._fingerprints()
        lock = Lock(
            version=1,
            digest_version=DIGEST_VERSION,
            digest=str(digests["digest"]),
            inputs=LockedGroup(
                path=os.path.relpath(
                    Path(input_file).absolute(), path.parent.absolute()
                ),
                digest=str(digests["inputs_digest"]),
                fingerprints=fps["inputs"],
            ),
            opts=LockedGroup(
                path=os.path.relpath(
                    Path(opts_file).absolute(), path.parent.absolute()
                ),
                digest=str(digests["opts_digest"]),
                fingerprints=fps["opts"],
            ),
        )
        publish_all(
            [
                (input_file, self._inputs_writer(inputs)),
                (opts_file, lambda p: _opts_json.write_opts(p, encoded)),
                (path, lock.write),
            ]
        )
        return path

    def load_lock(self, path: PathLike, /) -> tuple[dict[str, Any], dict[str, Any]]:
        """Verify and restore the exact pair; no extras or default filling."""
        path = Path(path)
        lock = read_lock(path)
        fps = self._fingerprints()
        if (
            lock.inputs.fingerprints != fps["inputs"]
            or lock.opts.fingerprints != fps["opts"]
        ):
            raise SchemaMismatchError(f"{path}: locked parameter schemas differ")
        input_path, opts_path = (
            path.parent / lock.inputs.path,
            path.parent / lock.opts.path,
        )
        f, stored_fps = _hdf5.open_checked(input_path, "inputs")
        with f:
            if stored_fps != lock.inputs.fingerprints:
                raise SchemaMismatchError(
                    f"{input_path}: fingerprints differ from lock"
                )
            inputs = self._read_inputs(f, input_path, stored_fps, "forbid", "raise")
        raw = _opts_json.read_opts(opts_path)
        self._resolve_names(
            "opts", opts_path, list(raw), None, self._spec.static, "forbid", "raise"
        )
        # Verify serialized values before validators can transform them.
        try:
            opts_digest = json_digest(self._ordered(self._spec.static, raw))
        except (TypeError, ValueError) as exc:
            raise StorageFormatError(
                f"{opts_path}: invalid serialized options"
            ) from exc
        inputs_digest = group_digest(self._spec.data, inputs)
        if inputs_digest != lock.inputs.digest or opts_digest != lock.opts.digest:
            raise DigestMismatchError(f"{path}: content differs from lock")
        if combined_digest(fps, inputs_digest, opts_digest) != lock.digest:
            raise DigestMismatchError(f"{path}: combined digest differs from lock")
        opts = {
            p.name: self._opts.decode_one(p, raw[p.name], str(opts_path))
            for p in self._spec.static
        }
        return inputs, opts

    # -- load ------------------------------------------------------------

    def _resolve_names(
        self,
        label: str,
        path: Path,
        stored: list[str],
        stored_fps: Mapping[str, Any] | None,
        params: tuple[Param, ...],
        extras: Extras,
        missing: Missing,
    ) -> tuple[list[Param], dict[str, Any]]:
        """Apply the extras/missing policy; return (params to read, filled defaults)."""
        if extras not in ("forbid", "ignore"):
            raise ValueError(f"extras must be 'forbid' or 'ignore', got {extras!r}")
        if missing not in ("raise", "default"):
            raise ValueError(f"missing must be 'raise' or 'default', got {missing!r}")

        expected = {p.name: p for p in params}
        for name in stored:
            if name in expected:
                continue
            other = self._spec.by_name.get(name)
            if other is not None:
                raise SchemaMismatchError(
                    f"{path}: stored parameter {name!r} is now role "
                    f"{other.role!r}, not part of {label}"
                )
            if extras == "forbid":
                raise SchemaMismatchError(
                    f"{path}: unknown stored name {name!r} (use extras='ignore' to drop)"
                )

        to_read: list[Param] = []
        filled: dict[str, Any] = {}
        stored_set = set(stored)
        for p in params:
            if p.name in stored_set:
                if stored_fps is not None and p.name not in stored_fps:
                    raise StorageFormatError(f"{path}: no fingerprint for {p.name!r}")
                if stored_fps is not None and stored_fps[p.name] != param_fingerprint(
                    p
                ):
                    raise SchemaMismatchError(
                        f"{path}: schema of {p.name!r} changed since the file was written"
                    )
                to_read.append(p)
            elif missing == "default" and p.has_default:
                filled[p.name] = p.default
            else:
                raise SchemaMismatchError(
                    f"{path}: missing stored value for {p.name!r}"
                )
        # Warn only once every name is known to resolve, so a raise stays silent.
        for name, default in filled.items():
            warnings.warn(
                f"{path}: {name!r} not stored; using current default {default!r}",
                stacklevel=4,
            )
        return to_read, filled

    def load_inputs(
        self,
        path: PathLike,
        /,
        *,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> dict[str, Any]:
        """Read and reconstruct the inputs file.

        Parameters
        ----------
        path
            File written by ``save_inputs`` or ``save``.
        extras
            ``"forbid"`` raises on stored names that are not parameters;
            ``"ignore"`` drops them.
        missing
            ``"raise"`` fails on parameters absent from the file; ``"default"``
            fills those that have a default and warns.

        Raises
        ------
        StorageFormatError
            If the file is malformed or has an unsupported version.
        SchemaMismatchError
            On role, name, or annotation mismatches.
        """
        path = Path(path)
        f, fps = _hdf5.open_checked(path, "inputs")
        with f:
            return self._read_inputs(f, path, fps, extras, missing)

    def _read_inputs(
        self,
        f: h5py.File,
        path: Path,
        fps: Mapping[str, Any],
        extras: Extras,
        missing: Missing,
    ) -> dict[str, Any]:
        group = f.get("inputs")
        if not isinstance(group, h5py.Group):
            raise StorageFormatError(f"{path}: missing /inputs group")
        to_read, filled = self._resolve_names(
            "inputs",
            path,
            list(group.keys()),
            fps,
            self._spec.data,
            extras,
            missing,
        )
        values = {
            p.name: _hdf5.read_node(group[p.name], _node(p), p.name) for p in to_read
        }
        return self._ordered(self._spec.data, values | filled)

    def load_opts(
        self,
        path: PathLike,
        /,
        *,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> dict[str, Any]:
        """Read and reconstruct the opts file; see ``load_inputs`` for policies."""
        path = Path(path)
        raw = _opts_json.read_opts(path)
        to_read, filled = self._resolve_names(
            "opts", path, list(raw), None, self._spec.static, extras, missing
        )
        values = {
            p.name: self._opts.decode_one(p, raw[p.name], str(path)) for p in to_read
        }
        return self._ordered(self._spec.static, values | filled)

    @staticmethod
    def _ordered(params: tuple[Param, ...], values: dict[str, Any]) -> dict[str, Any]:
        return {p.name: values[p.name] for p in params}

    def load(
        self,
        *,
        inputs: PathLike,
        opts: PathLike,
        skipped: Mapping[str, Any] | None = None,
        extras: Extras = "forbid",
        missing: Missing = "raise",
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Restore a legal ``(args, kwargs)`` pair without calling the callable.

        Parameters
        ----------
        inputs, opts
            Files written by ``save`` (or ``save_inputs`` / ``save_opts``).
        skipped
            Values for ``Skip`` parameters.
        extras, missing
            Policies, as in ``load_inputs``.
        """
        loaded_inputs = self.load_inputs(inputs, extras=extras, missing=missing)
        loaded_opts = self.load_opts(opts, extras=extras, missing=missing)
        merged = self._spec.merge(loaded_inputs, loaded_opts, skipped)
        return self._spec.to_call(merged)

    # -- output ----------------------------------------------------------

    def save_output(
        self,
        path: PathLike,
        output: R,
        /,
        *,
        inputs: Mapping[str, Any] | None = None,
        opts: Mapping[str, Any] | None = None,
    ) -> None:
        """Write ``output`` to HDF5 using the codec derived from the return annotation.

        Parameters
        ----------
        inputs, opts
            If given, content digests of these groups are recorded in the file.

        Raises
        ------
        OutputCodecError
            If no output codec can be derived from the return annotation.
        """
        node = self._spec.require_output()
        node.validate(output, "output")
        extra, _ = self._digests(inputs, opts)
        fps = {"output": output_fingerprint(node)}

        def write(p: Path) -> None:
            with h5py.File(p, "w") as f:
                _hdf5.write_envelope(f, "output", fps, **extra)
                _hdf5.write_node(f, "output", node, output)

        publish_all([(path, write)])

    def load_output(self, path: PathLike, /) -> R:
        """Read the output value written by ``save_output``.

        Raises
        ------
        OutputCodecError
            If no output codec can be derived from the return annotation.
        SchemaMismatchError
            If the stored output schema differs from the return annotation.
        """
        node = self._spec.require_output()
        path = Path(path)
        f, fps = _hdf5.open_checked(path, "output")
        with f:
            if fps.get("output") != output_fingerprint(node):
                raise SchemaMismatchError(
                    f"{path}: output schema differs from the return annotation"
                )
            if "output" not in f:
                raise StorageFormatError(f"{path}: missing /output")
            return _hdf5.read_node(f["output"], node, "output")


def fn[R](function: Callable[..., R], /) -> FunctionPersistence[R]:
    """Derive persistence methods from ``function``'s signature and annotations.

    Parameters
    ----------
    function
        Any annotated callable with a fixed signature. It is inspected, never run.

    Returns
    -------
    FunctionPersistence
        Generated ``flatten`` / ``save*`` / ``load*`` methods.

    Raises
    ------
    SpecError
        If a persisted parameter's annotation is unsupported or unresolvable.
    """
    return FunctionPersistence(function)


__all__ = ["Extras", "FunctionPersistence", "Missing", "fn"]
