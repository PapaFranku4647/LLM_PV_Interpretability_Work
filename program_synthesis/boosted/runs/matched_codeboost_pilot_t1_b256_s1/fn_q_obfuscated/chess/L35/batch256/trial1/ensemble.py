"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    g = lambda k: x.get(k, 'c0')
    return int(
        (g('x13') == 'c2' and (g('x11') != 'c0' or g('x19') == 'c1' or g('x29') == 'c1'))
        or (g('x19') == 'c1' and g('x8') == 'c1' and (g('x10') == 'c1' or g('x11') == 'c0'))
        or (g('x10') == 'c1' and g('x11') == 'c0' and g('x32') == 'c1' and g('x33') == 'c1')
        or (g('x0') == 'c1' and g('x22') == 'c1' and g('x24') == 'c1')
    )
h_1 = f

ALPHAS = [0.289038925387579]
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
