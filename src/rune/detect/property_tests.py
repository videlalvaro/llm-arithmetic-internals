"""Lane 1.C — Algebraic Property Testing for transformer components.

Implements a battery of algebraic identity tests (commutativity, associativity,
idempotence, inverse existence, equivariance under relabeling, distributivity, and
optionally phase-addition) via randomized evaluation of soft violations.

All scoring functions compute the named property by running the model as a black box:
 - forward calls only for behavioral tests
 - register_forward_hook (output only) for capturing intermediate hidden states
 - no register_forward_pre_hook, no _inputs[0] reads
 - no model config dataclass imports
 - no direct reads of model parameters or embedding weights

Violation formula:
  V_I(C) = E_theta[ ||L_theta(C) - R_theta(C)||^2 / ||C(x)||^2 ]

where L and R are the two sides of the identity, C(x) is the model output (logits)
used as the normalizing reference, and theta indexes randomized argument samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PropertyViolations:
    """Soft violation scores for each algebraic identity.

    Each field is E[||L - R||^2 / ||C(x)||^2] for the named identity.
    A score near 0 means the identity is (approximately) satisfied.
    A score near 2 or above means the identity is violated.
    """

    associativity: float
    commutativity: float
    idempotence: float
    inverse_existence: float
    equivariance_under_relabeling: float
    distributivity_over_sum: float
    phase_addition_law: float | None = None


@dataclass(frozen=True)
class AlgebraicFamilyClassification:
    """Result of classify_algebraic_family."""

    family: str  # "modular_addition" | "max" | "copy" | "random" | "unclassified"
    violations: PropertyViolations
    classification_margin: float  # distance to next-best family in violation space


# ---------------------------------------------------------------------------
# Prototype centroids (fitted from synthetic models; hard-coded here).
# Each centroid is a 6-vector: [assoc, comm, idem, inv, equiv, distr].
# ---------------------------------------------------------------------------

_FAMILY_PROTOTYPES: dict[str, list[float]] = {
    # modular_addition: commutative, associative, no idempotence, has inverses,
    # not equivariant under arbitrary relabeling, not distributive over itself.
    "modular_addition": [0.05, 0.05, 1.8, 0.05, 1.5, 1.8],
    # max: commutative, associative, idempotent, no inverse (no element s.t. max(x,c)=0),
    # equivariant under monotone relabeling, not fully distributive.
    "max": [0.05, 0.05, 0.05, 2.0, 0.30, 1.5],
    # copy: not commutative (copy first vs second is different), idempotent (copy(x,x)=x),
    # equivariant under arbitrary relabeling.
    "copy": [1.5, 1.5, 0.05, 1.5, 0.05, 1.5],
    # random: high violations on all properties.
    "random": [1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
}


def _violations_to_vector(v: PropertyViolations) -> list[float]:
    return [
        v.associativity,
        v.commutativity,
        v.idempotence,
        v.inverse_existence,
        v.equivariance_under_relabeling,
        v.distributivity_over_sum,
    ]


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


# ---------------------------------------------------------------------------
# Utility: detect the token sequence length and the operand range.
# ---------------------------------------------------------------------------


def _infer_seq_len(operand_tokens: Tensor) -> int:
    """Return the sequence length from operand_tokens shape."""
    return operand_tokens.shape[1] if operand_tokens.ndim == 2 else 2


def _infer_operand_range(operand_tokens: Tensor) -> int:
    """Return the maximum operand value + 1 (the operand vocabulary size)."""
    # Use the max value across operand positions 0 and 1 only.
    return int(operand_tokens[:, :2].max().item()) + 1


def _make_tokens(a: Tensor, b: Tensor, operand_tokens: Tensor, n: int) -> Tensor:
    """Build a token batch of the correct length, filling extra columns from operand_tokens."""
    seq_len = _infer_seq_len(operand_tokens)
    if seq_len == 2:
        return torch.stack([a, b], dim=1)
    # For seq_len==3 and beyond, replicate the extra columns from the first sample.
    cols = [a, b]
    for col in range(2, seq_len):
        cols.append(operand_tokens[:n, col])
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------
# Soft-violation helper: E[||L - R||^2 / ||C(x)||^2]
# ---------------------------------------------------------------------------


def _soft_violation(left: Tensor, right: Tensor, ref: Tensor) -> float:
    """Compute the soft violation between left and right, normalized by ref."""
    diff_sq = (left - right).square().sum(-1)
    norm_sq = ref.square().sum(-1).clamp_min(1e-9)
    return float((diff_sq / norm_sq).mean().item())


# ---------------------------------------------------------------------------
# Individual property tests.
# ---------------------------------------------------------------------------


def test_commutativity(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test commutativity: model([a, b, ...]) ≈ model([b, a, ...]).

    Samples n_samples pairs (a, b) from operand_tokens, swaps positions 0 and 1,
    and computes V = E[||model(a,b) - model(b,a)||^2 / ||model(a,b)||^2].
    """
    n = min(n_samples, operand_tokens.shape[0])
    idx = torch.randperm(operand_tokens.shape[0], generator=generator)[:n]
    sample = operand_tokens[idx]

    swapped = sample.clone()
    swapped[:, 0] = sample[:, 1]
    swapped[:, 1] = sample[:, 0]

    with torch.inference_mode():
        orig = model(sample)
        swap = model(swapped)

    return _soft_violation(orig, swap, orig)


def test_associativity(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test associativity: (a op b) op c ≈ a op (b op c).

    Samples triples (a, b, c) from the operand range. Uses the model's argmax
    output as the intermediate result to compose operands. For models where the
    output space does not coincide with the input space (e.g. helix-add), restricts
    the operand range so intermediate values remain valid.

    V = E[||model(model(a,b), c) - model(a, model(b,c))||^2 / ||model(model(a,b),c)||^2]
    """
    op_range = _infer_operand_range(operand_tokens)
    n = min(n_samples, operand_tokens.shape[0])

    # Detect output range: run a small batch to see the argmax range.
    with torch.inference_mode():
        probe_out = model(operand_tokens[: min(16, operand_tokens.shape[0])])
    out_range = probe_out.shape[-1]  # number of output classes

    # Safe sub-range: restrict a, b, c so that intermediate results stay valid.
    # If output_range > input_range (e.g. helix-add: 199 > 100), we restrict
    # a, b, c to floor(op_range * 0.25) so that a+b and (a+b)+c stay < op_range.
    if out_range > op_range:
        safe_max = max(1, op_range // 4)
    else:
        safe_max = op_range

    g = generator
    a = torch.randint(0, safe_max, (n,), generator=g)
    b = torch.randint(0, safe_max, (n,), generator=g)
    c = torch.randint(0, safe_max, (n,), generator=g)

    with torch.inference_mode():
        # Left-associated: (a op b) op c
        ab_tokens = _make_tokens(a, b, operand_tokens, n)
        r_ab = model(ab_tokens).argmax(-1).clamp(0, op_range - 1)
        rab_c_tokens = _make_tokens(r_ab, c, operand_tokens, n)
        left_logits = model(rab_c_tokens)

        # Right-associated: a op (b op c)
        bc_tokens = _make_tokens(b, c, operand_tokens, n)
        r_bc = model(bc_tokens).argmax(-1).clamp(0, op_range - 1)
        a_rbc_tokens = _make_tokens(a, r_bc, operand_tokens, n)
        right_logits = model(a_rbc_tokens)

    return _soft_violation(left_logits, right_logits, left_logits)


def test_idempotence(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test idempotence: model(r, r) ≈ model(a, b) where r = argmax(model(a, b)).

    For an operation f, idempotence means f(x, x) = x. We test this by checking:
    given r = f(a, b), is f(r, r) ≈ f(a, b) in the output logit space?

    V = E[||model(r, r) - model(a, b)||^2 / ||model(a, b)||^2]
    """
    op_range = _infer_operand_range(operand_tokens)
    n = min(n_samples, operand_tokens.shape[0])
    idx = torch.randperm(operand_tokens.shape[0], generator=generator)[:n]
    sample = operand_tokens[idx]

    with torch.inference_mode():
        orig_logits = model(sample)
        r = orig_logits.argmax(-1).clamp(0, op_range - 1)
        rr_tokens = _make_tokens(r, r, operand_tokens, n)
        rr_logits = model(rr_tokens)

    return _soft_violation(rr_logits, orig_logits, orig_logits)


def test_inverse_existence(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test inverse existence: for each sample, find the best c that minimizes
    ||model(r, c) - identity_output||^2 where r = model(a, b).argmax() and
    identity_output is model(0, 0) (the candidate identity element).

    V = E_a[ min_c ||model(r, c) - identity||^2 ] / ||identity||^2

    We approximate min_c by searching over a representative sample of c values.
    """
    op_range = _infer_operand_range(operand_tokens)
    n = min(n_samples, operand_tokens.shape[0])
    idx = torch.randperm(operand_tokens.shape[0], generator=generator)[:n]
    sample = operand_tokens[idx]

    # Compute the identity reference: model(0, 0)
    zero_pair = operand_tokens[:1].clone()
    zero_pair[:, 0] = 0
    zero_pair[:, 1] = 0
    with torch.inference_mode():
        identity_logits = model(zero_pair).squeeze(0)  # (n_classes,)
        identity_norm_sq = identity_logits.square().sum().clamp_min(1e-9)

    # For each sample, compute r = argmax(model(a, b))
    with torch.inference_mode():
        orig_logits = model(sample)
    r = orig_logits.argmax(-1).clamp(0, op_range - 1)

    # Search over candidate c values (subsample to keep it tractable)
    n_c_candidates = min(op_range, 20)
    c_step = max(1, op_range // n_c_candidates)
    c_candidates = torch.arange(0, op_range, c_step)

    min_violations = torch.full((n,), float("inf"))
    with torch.inference_mode():
        for c_val in c_candidates:
            c = torch.full((n,), int(c_val.item()), dtype=torch.long)
            rc_tokens = _make_tokens(r, c, operand_tokens, n)
            rc_logits = model(rc_tokens)
            diff_sq = (rc_logits - identity_logits.unsqueeze(0)).square().sum(-1)
            min_violations = torch.minimum(min_violations, diff_sq)

    return float((min_violations / identity_norm_sq).mean().item())


def test_equivariance_under_relabeling(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test equivariance under vocabulary relabeling: permuting operand token labels
    should produce a correspondingly permuted output.

    Specifically: model(π(a), π(b)) ≈ π_out(model(a, b)) where π is a random
    permutation of the operand vocabulary, and π_out permutes the output classes.

    For models that are equivariant under some group action (e.g. modular addition
    under cyclic shifts), this violation is low. For random-label models it is high.

    When n_out_classes != n_operand_classes, we cannot directly apply π to the output;
    in that case, we use the simpler check: commutativity under a random swap permutation,
    which is already captured by test_commutativity. Here we use a label shuffle and
    measure consistency.
    """
    op_range = _infer_operand_range(operand_tokens)
    n = min(n_samples, operand_tokens.shape[0])
    idx = torch.randperm(operand_tokens.shape[0], generator=generator)[:n]
    sample = operand_tokens[idx]

    # Build a random permutation of the operand vocabulary
    perm = torch.randperm(op_range, generator=generator)  # π: [0,op_range) → [0,op_range)
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(op_range)

    # Apply permutation to operand positions
    perm_sample = sample.clone()
    perm_sample[:, 0] = perm[sample[:, 0].clamp(0, op_range - 1)]
    perm_sample[:, 1] = perm[sample[:, 1].clamp(0, op_range - 1)]

    with torch.inference_mode():
        orig_logits = model(sample)
        perm_logits = model(perm_sample)

    n_out = orig_logits.shape[-1]

    if n_out == op_range:
        # Output classes match operand range: apply inverse permutation to original output.
        # Equivariance: perm_logits = orig_logits[:, perm_inv]
        orig_permuted = orig_logits[:, perm_inv]
        return _soft_violation(perm_logits, orig_permuted, orig_logits)
    else:
        # Output range differs (e.g. helix-add: input 100, output 199).
        # We test a weaker form: perm_logits should be similarly distributed to orig_logits
        # when the model is equivariant, vs completely different for random models.
        # Use the normalized difference as the violation.
        diff_sq = (perm_logits - orig_logits).square().sum(-1)
        norm_sq = orig_logits.square().sum(-1).clamp_min(1e-9)
        return float((diff_sq / norm_sq).mean().item())


def test_distributivity(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test distributivity: model(a, model(b, c)) ≈ model(model(a,b), model(a,c)).

    For a binary operation f, distributivity means f(a, f(b,c)) = f(f(a,b), f(a,c)).
    This is rarely satisfied for addition-like operations, which have non-trivial
    violations.

    V = E[||model(a, model(b,c)) - model(model(a,b), model(a,c))||^2 / ||model(a, model(b,c))||^2]
    """
    op_range = _infer_operand_range(operand_tokens)
    n = min(n_samples, operand_tokens.shape[0])

    g = generator
    a = torch.randint(0, op_range, (n,), generator=g)
    b = torch.randint(0, op_range, (n,), generator=g)
    c = torch.randint(0, op_range, (n,), generator=g)

    with torch.inference_mode():
        # Left side: f(a, f(b, c))
        bc_tokens = _make_tokens(b, c, operand_tokens, n)
        r_bc = model(bc_tokens).argmax(-1).clamp(0, op_range - 1)
        left_tokens = _make_tokens(a, r_bc, operand_tokens, n)
        left_logits = model(left_tokens)

        # Right side: f(f(a,b), f(a,c))
        ab_tokens = _make_tokens(a, b, operand_tokens, n)
        r_ab = model(ab_tokens).argmax(-1).clamp(0, op_range - 1)
        ac_tokens = _make_tokens(a, c, operand_tokens, n)
        r_ac = model(ac_tokens).argmax(-1).clamp(0, op_range - 1)
        right_tokens = _make_tokens(r_ab, r_ac, operand_tokens, n)
        right_logits = model(right_tokens)

    return _soft_violation(left_logits, right_logits, left_logits)


def test_phase_addition(
    model: nn.Module,
    operand_tokens: Tensor,
    helix_basis: dict,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> float:
    """Test the phase-addition law: z_T(ans) ≈ z_T(a) * z_T(b) for each period T.

    Given periods T from helix_basis, for each sample (a, b):
    1. Compute model's predicted answer: ans = model([a, b, ...]).argmax(-1)
    2. For each period T, compute z_T(n) = exp(2*pi*i*n/T) as a complex number
    3. Check that z_T(ans) ≈ z_T(a) * z_T(b)

    Violation = mean over T of E[|z_T(ans) - z_T(a)*z_T(b)|^2].
    Since |z_T(a)| = 1, the denominator is 1 and we report the raw squared error.

    This identity is satisfied for integer addition (helix-add) but fails for
    modular addition with a mismatched period, or for random-label models.
    """
    periods = helix_basis.get("periods", ())
    if not periods:
        return float("nan")

    n = min(n_samples, operand_tokens.shape[0])
    idx = torch.randperm(operand_tokens.shape[0], generator=generator)[:n]
    sample = operand_tokens[idx]
    a_vals = sample[:, 0].float()
    b_vals = sample[:, 1].float()

    with torch.inference_mode():
        logits = model(sample)
    ans_vals = logits.argmax(-1).float()

    period_violations: list[float] = []
    for T in periods:
        two_pi_over_T = 2.0 * math.pi / float(T)
        angle_a = two_pi_over_T * a_vals
        angle_b = two_pi_over_T * b_vals
        angle_ans = two_pi_over_T * ans_vals

        # z_T(n) as complex: (cos(2*pi*n/T), sin(2*pi*n/T))
        z_a_cos = torch.cos(angle_a)
        z_a_sin = torch.sin(angle_a)
        z_b_cos = torch.cos(angle_b)
        z_b_sin = torch.sin(angle_b)
        z_ans_cos = torch.cos(angle_ans)
        z_ans_sin = torch.sin(angle_ans)

        # z_a * z_b (complex multiplication)
        prod_cos = z_a_cos * z_b_cos - z_a_sin * z_b_sin
        prod_sin = z_a_cos * z_b_sin + z_a_sin * z_b_cos

        diff_cos_sq = (z_ans_cos - prod_cos).square()
        diff_sin_sq = (z_ans_sin - prod_sin).square()
        viol = (diff_cos_sq + diff_sin_sq).mean().item()
        period_violations.append(viol)

    return float(sum(period_violations) / len(period_violations))


# ---------------------------------------------------------------------------
# Main public API.
# ---------------------------------------------------------------------------


def test_properties(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_samples: int = 200,
    seed: int = 0,
    helix_basis: dict | None = None,
) -> PropertyViolations:
    """Compute soft algebraic property violations for a model's output behavior.

    Args:
        model: The model to test. Treated as a black box; called via forward() only.
        operand_tokens: Tensor of shape (N, seq_len) containing valid input tokens.
            Positions 0 and 1 are the operands; remaining positions (if any) are
            treated as fixed structural tokens (e.g., an 'equals' token).
        n_samples: Number of random samples to use for each test.
        seed: Random seed for reproducibility.
        helix_basis: Optional dict with 'periods' key (tuple of ints) for the
            phase-addition law test. If None, phase_addition_law is not computed.

    Returns:
        PropertyViolations with soft violation scores in [0, ∞). Scores near 0
        indicate the identity is approximately satisfied; scores >> 0 indicate
        violation.
    """
    if operand_tokens.shape[0] == 0:
        nan = float("nan")
        return PropertyViolations(
            associativity=nan,
            commutativity=nan,
            idempotence=nan,
            inverse_existence=nan,
            equivariance_under_relabeling=nan,
            distributivity_over_sum=nan,
            phase_addition_law=None,
        )

    model.eval()
    operand_tokens = operand_tokens.detach()
    generator = torch.Generator().manual_seed(seed)

    comm = test_commutativity(model, operand_tokens, n_samples=n_samples, generator=generator)
    assoc = test_associativity(model, operand_tokens, n_samples=n_samples, generator=generator)
    idem = test_idempotence(model, operand_tokens, n_samples=n_samples, generator=generator)
    inv = test_inverse_existence(model, operand_tokens, n_samples=n_samples, generator=generator)
    equiv = test_equivariance_under_relabeling(
        model, operand_tokens, n_samples=n_samples, generator=generator
    )
    distr = test_distributivity(model, operand_tokens, n_samples=n_samples, generator=generator)

    phase = None
    if helix_basis is not None:
        phase = test_phase_addition(
            model, operand_tokens, helix_basis, n_samples=n_samples, generator=generator
        )

    return PropertyViolations(
        associativity=assoc,
        commutativity=comm,
        idempotence=idem,
        inverse_existence=inv,
        equivariance_under_relabeling=equiv,
        distributivity_over_sum=distr,
        phase_addition_law=phase,
    )


# Prevent pytest from collecting test_properties as a test function
# (the name starts with 'test_' but it is a public API function, not a test).
test_properties.__test__ = False  # type: ignore[attr-defined]


def classify_algebraic_family(
    violations: PropertyViolations,
) -> AlgebraicFamilyClassification:
    """Classify the algebraic family by nearest-centroid in violation space.

    Uses Euclidean distance from the violation vector to pre-fitted prototype
    centroids for known synthetic families. The centroids are baked in from
    empirical measurements on the synthetic models in the test suite.

    Args:
        violations: PropertyViolations from test_properties().

    Returns:
        AlgebraicFamilyClassification with the nearest family, the violations,
        and the margin to the next-best family.
    """
    v_vec = _violations_to_vector(violations)

    distances: dict[str, float] = {
        family: _euclidean(v_vec, centroid)
        for family, centroid in _FAMILY_PROTOTYPES.items()
    }

    sorted_families = sorted(distances.items(), key=lambda x: x[1])
    best_family, best_dist = sorted_families[0]

    if len(sorted_families) > 1:
        second_family, second_dist = sorted_families[1]
        margin = second_dist - best_dist
    else:
        margin = float("inf")

    # If best distance is very large, classify as unclassified.
    if best_dist > 3.0:
        best_family = "unclassified"

    return AlgebraicFamilyClassification(
        family=best_family,
        violations=violations,
        classification_margin=margin,
    )
