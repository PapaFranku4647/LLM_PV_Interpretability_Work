"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Approximate hidden binary rule from the provided examples.
    Accepts a dict-like with keys x0..x7 or a sequence of length 8.
    Returns 0/1.
    """
    if isinstance(x, (list, tuple)):
        x0,x1,x2,x3,x4,x5,x6,x7 = x
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = (x[k] for k in ("x0","x1","x2","x3","x4","x5","x6","x7"))

    # High-curvature/low-x0 regime: almost always positive
    if x2 >= 3.0 or x3 <= -4.5:
        return 1

    # Large-x4 with weak drag (x6 near 0) tends to be positive
    if (x4 >= 40.0) and (x6 > -3.0) and (x2 >= 2.05):
        return 1

    # Moderate-x7, small x3, low x5 pocket that is often positive
    if (x0 >= 150.0) and (x7 > -90.0) and (x3 < 1.2) and (x5 < 14.0) and (x2 > 2.05):
        return 1

    return 0
h_1 = f

ALPHAS = [1.290949457887655]
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
