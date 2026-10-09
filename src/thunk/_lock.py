"""Strict metadata for a saved inputs/options pair."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ._errors import StorageFormatError

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class LockedGroup(_Model):
    path: str
    digest: Digest
    fingerprints: dict[str, Digest]
    representations: dict[str, Any]

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        if not value or "\x00" in value or Path(value).is_absolute():
            raise ValueError("path must be a nonempty relative path")
        return value


class Lock(_Model):
    version: Literal[2]
    digest_version: Literal[2]
    digest: Digest
    inputs: LockedGroup
    opts: LockedGroup

    @field_validator("version", "digest_version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_lock(path: Path) -> Lock:
    try:
        return Lock.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as exc:
        raise StorageFormatError(f"{path}: invalid lockfile: {exc}") from exc
