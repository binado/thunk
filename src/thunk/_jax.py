"""Optional JAX integration; importing this module never imports JAX."""

import sys
from typing import Any

import numpy as np

from ._errors import StorageFormatError, ValueTypeError

NUMERIC_DTYPES = frozenset(
    np.dtype(name)
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
)
X64_DTYPES = frozenset(
    np.dtype(n) for n in ("int64", "uint64", "float64", "complex128")
)
# Trailing key-data dimensions of JAX's built-in implementations.
KEY_SHAPES = {
    "threefry2x32": (2,),
    "threefry4x32": (4,),
    "philox2x32": (1,),
    "philox4x32": (2,),
    "rbg": (4,),
    "unsafe_rbg": (4,),
}


def is_annotation(tp: Any) -> bool:
    jax = sys.modules.get("jax")
    return jax is not None and tp is getattr(jax, "Array", None)


def validate(value: Any, path: str) -> str | None:
    """Validate without transferring data; return the key implementation, if any."""
    import jax
    from jax.core import Tracer

    if isinstance(value, Tracer):
        raise ValueTypeError(
            f"{path}: JAX tracers cannot be saved; persist outside jit/vmap/grad"
        )
    if not isinstance(value, jax.Array):
        raise ValueTypeError(
            f"{path}: expected jax.Array, got {type(value).__name__}; "
            "convert explicitly with jax.numpy.asarray"
        )
    if value.is_deleted():
        raise ValueTypeError(
            f"{path}: deleted JAX array; supply a live array before deletion or donation"
        )
    if not value.is_fully_addressable:
        raise ValueTypeError(
            f"{path}: JAX array is not fully addressable; gather it on this process before saving"
        )
    if getattr(value, "weak_type", False):
        raise ValueTypeError(
            f"{path}: weakly typed JAX array; specify an explicit dtype with jax.numpy.asarray"
        )
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        implementation = jax.random.key_impl(value)
        if (
            not isinstance(implementation, str)
            or implementation not in KEY_SHAPES
            or value.dtype != jax.random.key_dtype(implementation)
        ):
            raise ValueTypeError(
                f"{path}: unsupported PRNG implementation {implementation!r}; use a built-in implementation"
            )
        return implementation
    if value.dtype not in NUMERIC_DTYPES:
        raise ValueTypeError(
            f"{path}: unsupported JAX dtype {value.dtype}; cast to a standard bool, integer, float or complex dtype"
        )
    return None


def to_host(value: Any, path: str) -> tuple[np.ndarray, str | None]:
    """Shared, validated host encoding for storage and content hashing."""
    import jax

    implementation = validate(value, path)
    payload = jax.random.key_data(value) if implementation is not None else value
    return np.asarray(jax.device_get(payload)), implementation


def from_host(
    data: np.ndarray, kind: str, implementation: str | None, path: str
) -> Any:
    import jax

    if kind == "prng_key":
        if implementation not in KEY_SHAPES:
            raise StorageFormatError(
                f"{path}: unsupported PRNG implementation {implementation!r}"
            )
        key_shape = KEY_SHAPES[implementation]
        if (
            data.dtype != np.dtype("uint32")
            or data.shape[-len(key_shape) :] != key_shape
        ):
            raise StorageFormatError(
                f"{path}: malformed PRNG key payload; expected uint32 with trailing shape {key_shape}"
            )
        return jax.random.wrap_key_data(jax.device_put(data), dtype=implementation)
    if kind != "array":
        raise StorageFormatError(f"{path}: unsupported jax_kind {kind!r}")
    if data.dtype not in NUMERIC_DTYPES:
        raise StorageFormatError(f"{path}: unsupported JAX payload dtype {data.dtype}")
    if data.dtype in X64_DTYPES and not jax.config.x64_enabled:
        raise ValueTypeError(
            f"{path}: restoring {data.dtype} requires JAX X64; set JAX_ENABLE_X64=1 "
            "or jax.config.update('jax_enable_x64', True) before loading"
        )
    result = jax.device_put(data)
    if result.dtype != data.dtype or result.shape != data.shape:
        raise ValueTypeError(
            f"{path}: JAX changed dtype or shape during restoration; check JAX dtype configuration"
        )
    return result
