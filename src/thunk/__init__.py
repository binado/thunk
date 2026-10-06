"""Generate automatic save and load methods for functions that manipulate arrays"""

from importlib.metadata import version

from ._errors import (
    OutputCodecError,
    SchemaMismatchError,
    SerializerContractError,
    SpecError,
    StorageFormatError,
    ThunkError,
    ValueTypeError,
)
from ._markers import Data, DataSerializer, DataValidator, Skip, Static
from ._persistence import Extras, FunctionPersistence, Missing, fn

__version__ = version("thunk")

__all__ = [
    "Data",
    "DataSerializer",
    "DataValidator",
    "Extras",
    "FunctionPersistence",
    "Missing",
    "OutputCodecError",
    "SchemaMismatchError",
    "SerializerContractError",
    "Skip",
    "SpecError",
    "Static",
    "StorageFormatError",
    "ThunkError",
    "ValueTypeError",
    "__version__",
    "fn",
]
