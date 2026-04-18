"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic binary classifier for rows with features x0..x7.
    Accepts either a dict {'x0':..,'x1':..} or a sequence [x0..x7]."""
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (x[f'x{i}'] for i in range(8))
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = x

    # Strong signal: high x2 typically implies class 1
    if x2 >= 2.85:
        return 1

    # Low-x0 / low-x4 with very negative x6 tends to be class 0
    if x0 < 170 and x6 < -5.8 and x4 < 7.5:
        return 0

    # Small-x2, near-zero x7, positive x3 cases skew to class 0
    if x2 < 2.05 and x7 > -25 and x6 < -4.5 and x3 > 1.2:
        return 0

    # Linear score for the remaining ambiguous region
    score = (
        1.50 * x2 +
        0.02 * x4 +
        0.04 * x5 +
        0.25 * x6 +
        0.006 * x7 -
        0.006 * x0 -
        0.05 * x3
    )
    return 1 if score > 0.5 else 0
h_1 = f

ALPHAS = [1.048570559389618]
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
