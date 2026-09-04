"""Dependency-free normalization formulas used by the paper table builder."""

from __future__ import annotations


def _clip_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def regime_a_generation_retain(score: float, baseline: float) -> float:
    return _clip_percent(100.0 * score / baseline)


def regime_a_generation_forget(score: float, baseline: float) -> float:
    return _clip_percent(100.0 * (1.0 - score / baseline))


def regime_a_mcq_retain(score: float, baseline: float, chance: float = 0.25) -> float:
    return _clip_percent(100.0 * (score - chance) / (baseline - chance))


def regime_a_mcq_forget(score: float, baseline: float, chance: float = 0.25) -> float:
    return _clip_percent(100.0 * (1.0 - (score - chance) / (baseline - chance)))


def regime_b_generation(
    retain: float,
    forget: float,
    pooled_baseline: float,
) -> tuple[float, float, float]:
    retain_score = 100.0 * retain / pooled_baseline
    forget_score = 100.0 * forget / pooled_baseline
    return retain_score, forget_score, forget_score - retain_score


def regime_b_mcq(
    retain: float,
    forget: float,
    pooled_baseline: float,
    chance: float = 0.25,
) -> tuple[float, float, float]:
    retain_score = 100.0 * (retain - chance) / (pooled_baseline - chance)
    forget_score = 100.0 * (forget - chance) / (pooled_baseline - chance)
    return retain_score, forget_score, forget_score - retain_score
