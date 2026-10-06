"""Atomic file publication via a temporary file in the destination directory."""

import contextlib
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

type Writer = Callable[[Path], None]


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _new_temp(dest: Path) -> Path:
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    return Path(name)


def _publish(temp: Path, dest: Path) -> None:
    try:
        mode = dest.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o666 & ~_umask()
    os.chmod(temp, mode)
    os.replace(temp, dest)


def atomic_write(path: os.PathLike[str] | str, writer: Writer) -> None:
    publish_all([(path, writer)])


def publish_all(items: Sequence[tuple[os.PathLike[str] | str, Writer]]) -> None:
    """Encode every item to a temp file, then replace the destinations.

    No destination is touched unless every writer succeeds. The replacements
    themselves are sequential, so they are not one transaction.
    """
    dests = [Path(p) for p, _ in items]
    resolved = [d.resolve() for d in dests]
    if len(set(resolved)) != len(resolved):
        raise ValueError("destination paths must be distinct")

    temps: list[Path] = []
    try:
        for dest, (_, writer) in zip(dests, items, strict=True):
            temp = _new_temp(dest)
            temps.append(temp)
            writer(temp)
        for temp, dest in zip(temps, dests, strict=True):
            _publish(temp, dest)
    finally:
        for temp in temps:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()
