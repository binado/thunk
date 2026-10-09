"""Exception hierarchy for thunk."""


class ThunkError(Exception):
    """Base class for every error raised deliberately by thunk."""


class SpecError(ThunkError, TypeError):
    """A callable or annotation cannot be compiled into a persistence spec."""


class ValueTypeError(ThunkError, TypeError):
    """A runtime value violates a storage contract."""


class SchemaMismatchError(ThunkError, ValueError):
    """Stored names, roles, or fingerprints disagree with the current callable."""


class StorageFormatError(ThunkError, ValueError):
    """A file is not a valid thunk file or has an unsupported storage version."""


class SerializerContractError(ThunkError, ValueError):
    """A custom serializer or validator violated its contract."""


class OutputCodecError(ThunkError, TypeError):
    """The output cannot be persisted with its storage contract."""


class DigestMismatchError(ThunkError, ValueError):
    """Stored content or the combined key disagrees with its lockfile or cache key."""


class ReconstructionError(ThunkError, ValueError):
    """An applicable dataclass reconstruction hint failed in its constructor."""
