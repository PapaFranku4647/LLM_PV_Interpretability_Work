"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule-based classifier for rows with keys x0..x7."""
    x0,x1,x2,x3,x4,x5,x6,x7 = (x[k] for k in ("x0","x1","x2","x3","x4","x5","x6","x7"))

    return int(
        (x3 <= -6.0) or
        (x7 <= 2.1 and x3 <= -1.5 and x5 >= 25.0) or
        (x4 <= -20.0 and x6 <= 0.5) or
        (x3 <= -3.0 and x5 >= 12.0 and x1 >= 95.0 and x7 <= 6.5)
    )
h_1 = f

ALPHAS = [1.009668808805065]
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
