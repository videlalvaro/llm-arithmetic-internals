"""Neural Hankel tomography utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Symbol = TypeVar("Symbol")
SequenceKey = tuple[Symbol, ...]
Component = Callable[[SequenceKey[Symbol]], object]
Readout = Callable[[object], float]


@dataclass(frozen=True)
class RankConfidence:
    rank: int
    low: float
    high: float
    samples: tuple[int, ...]


@dataclass(frozen=True)
class RankGrowthResult:
    ranks: tuple[int, ...]
    stable_at: int | None
    block: NDArray[np.float64]


@dataclass(frozen=True)
class HankelCandidate:
    component_name: str
    rank: int
    score: float
    block: NDArray[np.float64]


def compute_hankel_block(
    component: Component[Symbol],
    readout: Readout,
    prefixes: Sequence[SequenceKey[Symbol]],
    suffixes: Sequence[SequenceKey[Symbol]],
) -> NDArray[np.float64]:
    """Build H[p, s] = readout(component(p + s))."""

    block = np.empty((len(prefixes), len(suffixes)), dtype=np.float64)
    for row, prefix in enumerate(prefixes):
        for column, suffix in enumerate(suffixes):
            block[row, column] = readout(component(tuple(prefix) + tuple(suffix)))
    return block


def stable_rank(matrix: NDArray[np.float64], epsilon: float = 1e-6) -> int:
    if matrix.ndim != 2:
        raise ValueError("stable_rank expects a matrix")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    threshold = epsilon * max(matrix.shape) * singular_values[0]
    return int(np.sum(singular_values > threshold))


def bootstrap_rank_confidence(
    matrix: NDArray[np.float64],
    *,
    epsilon: float = 1e-6,
    samples: int = 128,
    seed: int = 0,
    confidence: float = 0.95,
) -> RankConfidence:
    if matrix.ndim != 2:
        raise ValueError("bootstrap_rank_confidence expects a matrix")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between 0 and 1")

    generator = np.random.default_rng(seed)
    ranks: list[int] = []
    rows, columns = matrix.shape
    for _ in range(samples):
        row_index = generator.integers(0, rows, size=rows)
        column_index = generator.integers(0, columns, size=columns)
        ranks.append(stable_rank(matrix[np.ix_(row_index, column_index)], epsilon=epsilon))

    alpha = (1.0 - confidence) / 2.0
    return RankConfidence(
        rank=stable_rank(matrix, epsilon=epsilon),
        low=float(np.quantile(ranks, alpha)),
        high=float(np.quantile(ranks, 1.0 - alpha)),
        samples=tuple(ranks),
    )


def binary_strings(max_length: int) -> tuple[tuple[int, ...], ...]:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    strings: list[tuple[int, ...]] = []
    for length in range(max_length + 1):
        for value in range(2**length):
            strings.append(tuple((value >> shift) & 1 for shift in reversed(range(length))))
    return tuple(strings)


def adaptive_hankel_growth(
    component: Component[Symbol],
    readout: Readout,
    alphabet: Sequence[Symbol],
    *,
    max_length: int,
    epsilon: float = 1e-6,
    patience: int = 2,
) -> RankGrowthResult:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if patience < 1:
        raise ValueError("patience must be at least 1")

    ranks: list[int] = []
    stable_at: int | None = None
    final_block = np.empty((0, 0), dtype=np.float64)
    for length in range(max_length + 1):
        strings = bounded_strings(alphabet, length)
        final_block = compute_hankel_block(component, readout, strings, strings)
        ranks.append(stable_rank(final_block, epsilon=epsilon))
        if len(ranks) > patience and len(set(ranks[-(patience + 1) :])) == 1:
            stable_at = length
            break
    return RankGrowthResult(ranks=tuple(ranks), stable_at=stable_at, block=final_block)


def scan_components(
    components: Mapping[str, Component[Symbol]],
    readout: Readout,
    prefixes: Sequence[SequenceKey[Symbol]],
    suffixes: Sequence[SequenceKey[Symbol]],
    *,
    epsilon: float = 1e-6,
) -> tuple[HankelCandidate, ...]:
    candidates = []
    for name, component in components.items():
        block = compute_hankel_block(component, readout, prefixes, suffixes)
        rank = stable_rank(block, epsilon=epsilon)
        candidates.append(
            HankelCandidate(
                component_name=name,
                rank=rank,
                score=float(rank),
                block=block,
            )
        )
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.score, candidate.component_name))
    )


def bounded_strings(
    alphabet: Sequence[Symbol],
    max_length: int,
) -> tuple[tuple[Symbol, ...], ...]:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    strings: list[tuple[Symbol, ...]] = [()]
    frontier: list[tuple[Symbol, ...]] = [()]
    for _ in range(max_length):
        frontier = [prefix + (symbol,) for prefix in frontier for symbol in alphabet]
        strings.extend(frontier)
    return tuple(strings)