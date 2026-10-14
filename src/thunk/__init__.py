"""Generate automatic save and load methods for functions that manipulate arrays"""

from importlib.metadata import version

from ._errors import (
    DigestMismatchError,
    OutputCodecError,
    ReconstructionError,
    SchemaMismatchError,
    SerializerContractError,
    SpecError,
    StorageFormatError,
    ThunkError,
    ValueTypeError,
)
from ._markers import (
    Data,
    DataDeserializer,
    DataSerializer,
    DataValidator,
    Skip,
    Static,
)
from ._persistence import Extras, FunctionPersistence, Missing, cache, fn

__version__ = version("thunk")

__all__ = [
    "Data",
    "DataSerializer",
    "DataDeserializer",
    "DataValidator",
    "DigestMismatchError",
    "Extras",
    "FunctionPersistence",
    "Missing",
    "OutputCodecError",
    "ReconstructionError",
    "SchemaMismatchError",
    "SerializerContractError",
    "Skip",
    "SpecError",
    "Static",
    "StorageFormatError",
    "ThunkError",
    "ValueTypeError",
    "__version__",
    "cache",
    "fn",
]
