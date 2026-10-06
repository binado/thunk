"""Role markers and custom-codec wrappers used inside ``Annotated``."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import PlainSerializer, PlainValidator

from ._errors import SpecError


@dataclass(frozen=True)
class DataSerializer:
    """Turn a value into a nested ``dict[str, dict | ndarray | scalar]``."""

    func: Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class DataValidator:
    """Rebuild a value from the nested dictionary a ``DataSerializer`` produced."""

    func: Callable[[dict[str, Any]], Any]


def _both_or_neither(marker: str, serializer: object, validator: object) -> None:
    if (serializer is None) != (validator is None):
        raise SpecError(f"{marker}: serializer and validator must be given together")


@dataclass(frozen=True)
class Data:
    """Persist the parameter in the HDF5 inputs file."""

    serializer: DataSerializer | None = None
    validator: DataValidator | None = None

    def __post_init__(self) -> None:
        _both_or_neither("Data", self.serializer, self.validator)
        if self.serializer is not None and not isinstance(
            self.serializer, DataSerializer
        ):
            raise SpecError("Data.serializer must be a thunk.DataSerializer")
        if self.validator is not None and not isinstance(self.validator, DataValidator):
            raise SpecError("Data.validator must be a thunk.DataValidator")


@dataclass(frozen=True)
class Static:
    """Persist the parameter in the JSON opts file."""

    serializer: PlainSerializer | None = None
    validator: PlainValidator | None = None

    def __post_init__(self) -> None:
        _both_or_neither("Static", self.serializer, self.validator)
        if self.serializer is not None:
            if not isinstance(self.serializer, PlainSerializer):
                raise SpecError("Static.serializer must be a pydantic PlainSerializer")
            if self.serializer.when_used != "always":
                raise SpecError("Static.serializer must use when_used='always'")
        if self.validator is not None and not isinstance(
            self.validator, PlainValidator
        ):
            raise SpecError("Static.validator must be a pydantic PlainValidator")


@dataclass(frozen=True)
class Skip:
    """Leave the parameter out of storage; supply it at call time instead."""


ROLE_MARKERS = (Data, Static, Skip)
