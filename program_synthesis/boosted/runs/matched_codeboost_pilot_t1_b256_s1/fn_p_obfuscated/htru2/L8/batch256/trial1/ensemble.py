"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule for binary label from 8 features x0..x7.
    Accepts x as a sequence (x[0]..x[7]) or a dict with keys 'x0'..'x7'."""
    if isinstance(x, dict):
        x0, x2, x3, x4 = x['x0'], x['x2'], x['x3'], x['x4']
    else:
        x0, x2, x3, x4 = x[0], x[2], x[3], x[4]

    return int(
        (x2 >= 2.95) or              # high x2 cluster
        (x3 <= -6.0) or              # strongly negative x3 cluster
        (x0 < 130 and x3 < -0.5) or  # very low x0 with negative x3
        (x0 < 180 and x4 > 60 and x3 < 1.0)  # high x4 (relative) at moderate x0
    )
h_1 = f

ALPHAS = [1.157772118818275]
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
