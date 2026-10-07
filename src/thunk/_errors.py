"""Exception hierarchy for thunk."""


class ThunkError(Exception):
    """Base class for every error raised deliberately by thunk."""


class SpecError(ThunkError, TypeError):
    """A callable or annotation cannot be compiled into a persistence spec."""


class ValueTypeError(ThunkError, TypeError):
    """A runtime value does not match its annotation-derived spec."""


class SchemaMismatchError(ThunkError, ValueError):
    """Stored names, roles, or fingerprints disagree with the current callable."""


class StorageFormatError(ThunkError, ValueError):
    """A file is not a valid thunk file or has an unsupported storage version."""


class SerializerContractError(ThunkError, ValueError):
    """A custom serializer or validator violated its contract."""


class OutputCodecError(ThunkError, TypeError):
    """No supported output codec can be derived from the return annotation."""


class DigestMismatchError(ThunkError, ValueError):
    """Stored content or the combined key disagrees with its lockfile."""
