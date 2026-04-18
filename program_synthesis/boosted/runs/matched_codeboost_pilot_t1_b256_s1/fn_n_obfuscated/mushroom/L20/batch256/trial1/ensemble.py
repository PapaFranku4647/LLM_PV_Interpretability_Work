"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule-based classifier for the provided schema.
    Expects x as a dict-like object with keys 'x0'..'x19'.
    Numeric features: x0, x8, x9 (floats). Others are categorical strings like 'c07'.
    Returns 0/1.
    """
    def g(k, default=None):
        return x.get(k, default) if hasattr(x, 'get') else x[k]

    def as_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    x0 = as_float(g('x0', 0.0))
    x8 = as_float(g('x8', 0.0))
    x9 = as_float(g('x9', 0.0))

    x1 = g('x1', None)
    x3 = g('x3', None)
    x5 = g('x5', None)
    x6 = g('x6', None)
    x7 = g('x7', None)
    x11 = g('x11', None)
    x14 = g('x14', None)

    # Strong negative cluster in the samples
    if x6 == 'c2' and x7 == 'c02':
        return 0

    # Very large x0 is consistently positive in the examples
    if x0 > 2.5:
        return 1

    # Some clearly negative cases with x1=c0 and x5=c4
    if x1 == 'c0' and x5 == 'c4' and x9 < -1.3:
        return 0

    # Main numeric decision boundary
    if x9 <= -1.6:
        return 1

    # Secondary: very low x8 together with moderately low x9
    if x8 <= -3.2 and x9 <= -1.0:
        return 1

    # Category-driven boosts seen frequently with positives
    if x1 == 'c2' and x9 <= -0.8:
        return 1

    if (x11 == 'c8' or x14 == 'c5') and x9 <= -1.0:
        return 1

    # Mild guard: some patterns with x3=c07 skew negative
    if x3 == 'c07' and x9 > -1.4:
        return 0

    return 0
h_1 = f

ALPHAS = [0.4217901919935356]
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
