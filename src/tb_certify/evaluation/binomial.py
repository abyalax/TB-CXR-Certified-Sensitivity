"""Exact binomial confidence intervals for audit reporting."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class BinomialInterval:
    successes: int
    trials: int
    confidence: float
    lower: float
    upper: float


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> BinomialInterval:
    """Two-sided Clopper-Pearson interval, including exact boundary cases."""

    if trials < 1:
        raise ValueError("trials must be at least one.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one.")

    tail_probability = (1 - confidence) / 2
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail_probability, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - tail_probability, successes + 1, trials - successes))
    )
    return BinomialInterval(
        successes=successes,
        trials=trials,
        confidence=confidence,
        lower=lower,
        upper=upper,
    )
