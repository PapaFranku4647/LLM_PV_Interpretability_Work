"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Binary rule approximating the hidden target.
    Accepts x as a sequence [x0..x7] or a dict with keys 'x0'..'x7'.
    Returns 0/1.
    """
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (x[f'x{i}'] for i in range(8))
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = x

    # Main regime: larger x2 together with non-trivial x5 tends to class 1
    if (x2 > 2.6) and (x5 > 15):
        return 1

    # Secondary regime: occasionally class 1 even with smaller x2 when x4/x5 are large
    # and the tail (x7) is not very negative.
    if (x2 > 1.65) and (x4 < 30) and (x5 > 30) and (x7 > -20) and (x6 > -5) and (x1 > -30):
        return 1

    return 0
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
