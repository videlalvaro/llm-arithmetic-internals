"""Mechanism plethora discovery helpers.

This is the first OASR-style wrapper layer: it does not implement a base detector
itself, but repeatedly asks a base detector for candidate realizations and adds
overlap repulsion against already accepted candidates.

Algorithm-family repulsion (``algorithm_family_repulsion=True`` in
``discover_plethora``) is a second repulsion pass that runs AFTER the
subspace-overlap pass.  It prevents the wrapper from accepting two candidates
that belong to the same algorithm family.

Family classification:
  Each ``CandidateRealization`` may carry an optional ``metadata['family']``
  string (preferred) or ``metadata['algorithm']`` fallback.  The set of
  allowed well-known family tags is listed in ``_KNOWN_ALGORITHM_FAMILIES``.
  Candidates without a family tag are classified as ``'generic_neural'``.
  Unrecognized tags compare by string equality (treated as opaque symbols),
  which preserves forward-compatibility when new families are added.

Calibration note for ``_KNOWN_ALGORITHM_FAMILIES``:
  Populated from the Kantamneni-Tegmark taxonomy of Fourier / lookup / helix
  mechanisms for modular arithmetic plus the helix-clock family discovered in
  this project.  Tags are lower-case kebab-free identifiers; the set is
  intentionally open (see ``repel_algorithm_family`` docstring).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Named module-level constants
# ---------------------------------------------------------------------------

# Canonical well-known algorithm-family tags.
# Calibration: derived from Kantamneni & Tegmark (2023) taxonomy of modular-
# arithmetic circuits plus the families tracked in this project's MDL module
# (see src/rune/detect/mdl.py ``_FAMILY_*`` constants).
# This set is informational only — ``repel_algorithm_family`` accepts ANY
# string tag and compares by equality for tags outside this set.
_KNOWN_ALGORITHM_FAMILIES: frozenset[str] = frozenset({
    "clock",
    "modular_affine",
    "lookup",
    "digit_collation",
    "wfa",
    "helix",
    "generic_neural",
})


@dataclass(frozen=True)
class CandidateRealization:
    """A detector-emitted candidate neural realization for one semantics."""

    id: str
    score: float
    edges: frozenset[str] = frozenset()
    nodes: frozenset[str] = frozenset()
    subspace_basis: NDArray[np.float64] | None = None
    replace_kl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlethoraProfile:
    """Set of low-overlap candidate realizations plus pairwise overlap matrices."""

    candidates: tuple[CandidateRealization, ...]
    edge_iou: NDArray[np.float64]
    node_iou: NDArray[np.float64]
    subspace_overlap: NDArray[np.float64]

    @property
    def cardinality(self) -> int:
        return len(self.candidates)


BaseDetector = Callable[[Sequence[CandidateRealization]], Iterable[CandidateRealization]]


def set_iou(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def projection_overlap(
    left: NDArray[np.float64] | None,
    right: NDArray[np.float64] | None,
) -> float:
    """Return normalized projection overlap for two column-basis matrices."""

    if left is None or right is None:
        return 0.0
    left_projection = left @ np.linalg.pinv(left)
    right_projection = right @ np.linalg.pinv(right)
    numerator = np.linalg.norm(left_projection @ right_projection, ord="fro") ** 2
    denominator = np.linalg.norm(left_projection, ord="fro") * np.linalg.norm(
        right_projection,
        ord="fro",
    )
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def candidate_overlap(left: CandidateRealization, right: CandidateRealization) -> float:
    return max(
        set_iou(left.edges, right.edges),
        projection_overlap(left.subspace_basis, right.subspace_basis),
    )


def repelled_score(
    candidate: CandidateRealization,
    accepted: Sequence[CandidateRealization],
    *,
    lambda_repel: float,
) -> float:
    return candidate.score + lambda_repel * sum(
        candidate_overlap(candidate, previous) for previous in accepted
    )


def _get_family_tag(candidate: CandidateRealization) -> str:
    """Extract the algorithm-family tag from a candidate's metadata.

    Preference order: ``metadata['family']`` → ``metadata['algorithm']`` →
    ``'generic_neural'`` (fallback when neither key is present).

    The tag is returned as-is; callers are responsible for equality comparisons.
    We do not normalise casing or strip whitespace so that the classification
    is purely data-driven and unrecognised tags remain distinguishable.
    """
    family = candidate.metadata.get("family")
    if family is not None:
        return str(family)
    algorithm = candidate.metadata.get("algorithm")
    if algorithm is not None:
        return str(algorithm)
    return "generic_neural"


def repel_algorithm_family(
    accepted_realizations: tuple[CandidateRealization, ...],
    candidate: CandidateRealization,
) -> bool:
    """Return True if ``candidate`` represents a DIFFERENT algorithm family than
    any accepted realization, False if it duplicates an algorithm already
    accepted.

    Algorithm-family classification uses the candidate's ``metadata['family']``
    string if present, falling back to ``metadata['algorithm']``.  Allowed
    well-known family tags (see ``_KNOWN_ALGORITHM_FAMILIES``): ``"clock"``,
    ``"modular_affine"``, ``"lookup"``, ``"digit_collation"``, ``"wfa"``,
    ``"helix"``, ``"generic_neural"``.  Unrecognized tags compare by string
    equality, so they are treated as distinct families unless two candidates
    share the exact same tag string.

    A candidate with no accepted_realizations is trivially new (returns True).

    The function is intentionally stateless: pass the current accepted set on
    each call; do not mutate any global state.
    """
    if not accepted_realizations:
        return True
    candidate_family = _get_family_tag(candidate)
    accepted_families = {_get_family_tag(r) for r in accepted_realizations}
    # The candidate is NEW (different) iff its family has NOT been seen yet.
    return candidate_family not in accepted_families


def discover_plethora(
    base_detector: BaseDetector,
    *,
    max_realizations: int = 8,
    score_threshold: float = float("inf"),
    lambda_repel: float = 1.0,
    algorithm_family_repulsion: bool = False,
) -> PlethoraProfile:
    """Sequentially discover low-overlap realizations from a base detector.

    Parameters
    ----------
    base_detector:
        Callable ``(accepted_so_far) -> Iterable[CandidateRealization]``.
        Called on each iteration with the accepted set so far.
    max_realizations:
        Hard upper bound on the number of accepted realizations.
    score_threshold:
        Reject a candidate whose raw ``score`` exceeds this value.
    lambda_repel:
        Weight for the subspace-overlap repulsion penalty in
        ``repelled_score``.
    algorithm_family_repulsion:
        When ``True``, after the subspace-overlap repulsion pass also
        apply ``repel_algorithm_family`` against the accepted set.  A
        candidate is admitted only if it represents a family NOT already
        in the accepted set.  Defaults to ``False`` to preserve the
        existing behaviour of the wrapper.
    """

    accepted: list[CandidateRealization] = []
    accepted_ids: set[str] = set()
    for _ in range(max_realizations):
        candidates = [
            candidate
            for candidate in base_detector(tuple(accepted))
            if candidate.id not in accepted_ids
        ]
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda candidate: repelled_score(
                candidate,
                accepted,
                lambda_repel=lambda_repel,
            ),
        )
        if selected.score > score_threshold:
            break
        # Algorithm-family repulsion: skip candidates from an already-accepted
        # family.  The subspace-overlap score may still pick a same-family
        # candidate as the best-scoring one, so we filter AFTER score-ranking.
        if algorithm_family_repulsion and not repel_algorithm_family(
            tuple(accepted), selected
        ):
            # The best candidate duplicates an existing family.  Try to find
            # the best candidate from a NEW family instead.
            new_family_candidates = [
                c
                for c in candidates
                if repel_algorithm_family(tuple(accepted), c)
            ]
            if not new_family_candidates:
                # No new-family candidate available; stop discovery.
                break
            selected = min(
                new_family_candidates,
                key=lambda candidate: repelled_score(
                    candidate,
                    accepted,
                    lambda_repel=lambda_repel,
                ),
            )
            if selected.score > score_threshold:
                break
        accepted.append(selected)
        accepted_ids.add(selected.id)

    return build_plethora_profile(accepted)


def build_plethora_profile(candidates: Sequence[CandidateRealization]) -> PlethoraProfile:
    count = len(candidates)
    edge_iou = np.zeros((count, count), dtype=np.float64)
    node_iou = np.zeros((count, count), dtype=np.float64)
    subspace = np.zeros((count, count), dtype=np.float64)
    for row, left in enumerate(candidates):
        for column, right in enumerate(candidates):
            edge_iou[row, column] = set_iou(left.edges, right.edges)
            node_iou[row, column] = set_iou(left.nodes, right.nodes)
            subspace[row, column] = projection_overlap(left.subspace_basis, right.subspace_basis)
    return PlethoraProfile(
        candidates=tuple(candidates),
        edge_iou=edge_iou,
        node_iou=node_iou,
        subspace_overlap=subspace,
    )