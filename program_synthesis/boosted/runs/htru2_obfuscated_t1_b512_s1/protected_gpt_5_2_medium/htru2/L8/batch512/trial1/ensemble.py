"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule-based classifier for inputs x0..x7.
    Accepts either a length-8 sequence or a dict with keys 'x0'..'x7'.
    Returns 0/1.
    """
    if isinstance(x, dict):
        x0, x1, x2, x3, x4, x5, x6, x7 = (x[f"x{i}"] for i in range(8))
    else:
        x0, x1, x2, x3, x4, x5, x6, x7 = x

    # Strong positive region: large negative x3 (many positives)
    if x3 < -4.0:
        return 1

    # Strong positive region: larger x2 (many positives even when other signals disagree)
    if x2 > 2.85:
        return 1

    # Strong negative region: small x2 with clearly positive x3 (dominant negative cluster)
    if x2 < 2.15 and x3 > 0.9:
        return 0

    # Additional negative patterns (high x0 + small x2 + positive x3)
    if x0 > 185.0 and x2 < 2.25 and x3 > 0.3:
        return 0

    # Very negative x6 with small x4 often indicates the negative cluster
    if x6 < -9.0 and x4 < 8.0 and x2 < 2.55 and x3 > -1.0:
        return 0

    # Extremely negative x7 with small x4 and small x2 tends to be negative
    if x4 < 6.0 and x2 < 2.4 and x7 < -80.0:
        return 0

    return 1
h_1 = f

ALPHAS = [1.261810580984345]
LEARNERS = [h_1]

def _normalize_pred_to_pm1(pred: Any) -> int:
    try:
        if hasattr(pred, 'item'):
            pred = pred.item()
    except Exception:
        pass
    if isinstance(pred, bool):
        return 1 if pred else -1
    if isinstance(pred, int):
        return 1 if pred != 0 else -1
    if isinstance(pred, str):
        s = pred.strip().strip("\"'")
        if s in ('1', 'true', 'True'):
            return 1
        if s in ('0', 'false', 'False', ''):
            return -1
        try:
            return 1 if int(float(s)) != 0 else -1
        except Exception:
            return 1 if s else -1
    return 1 if pred else -1

def f(x: Any) -> int:
    score = 0.0
    for alpha, learner in zip(ALPHAS, LEARNERS):
        score += alpha * _normalize_pred_to_pm1(learner(x))
    return 1 if score >= 0.0 else 0
