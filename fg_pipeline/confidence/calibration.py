from __future__ import annotations

import math


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    """Keep confidence values inside a numerically safe probability range."""

    return max(eps, min(1.0 - eps, value))


def temperature_scale(probability: float, temperature: float = 1.0) -> float:
    """Simple binary temperature scaling helper for calibrated confidence."""

    probability = clamp_probability(probability)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logit = math.log(probability / (1.0 - probability))
    scaled = 1.0 / (1.0 + math.exp(-(logit / temperature)))
    return clamp_probability(scaled)
