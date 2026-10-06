"""Callables shared by tests, importable from a fresh process."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np

import thunk


@dataclass(frozen=True)
class Population:
    positions: np.ndarray
    labels: list[str]
    scale: float = 1.0


@dataclass
class Config:
    name: str
    sizes: tuple[int, ...]
    ratio: float
    mode: str | None = None


def spectra(
    seeds: np.ndarray,
    /,
    population: Population,
    *,
    bins: int = 128,
    chunk_size: Annotated[int, thunk.Skip()] = 4096,
) -> np.ndarray:
    return seeds * bins + population.positions.sum() + chunk_size


def explode(
    x: np.ndarray, /, n: int = 1, *, tag: Annotated[str, thunk.Skip()] = "t"
) -> np.ndarray:
    raise RuntimeError("body must not run when restoring arguments")
