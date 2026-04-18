"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    x0 = x['x0']; x2 = x['x2']; x4 = x['x4']; x6 = x['x6']
    # Main separation: large x2 almost always corresponds to class 1
    if x2 >= 2.7:
        return 1
    # Many low-x0 cases are class 1 when x4 is not tiny
    if x0 <= 160 and x4 >= 15:
        return 1
    # A smaller pocket of class 1: moderate x0 with very large x4 and mild x6
    if x4 >= 55 and x0 <= 200 and x2 >= 1.8 and x6 > -3.2:
        return 1
    return 0
h_1 = f

ALPHAS = [1.354025100551105]
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
