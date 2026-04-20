"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule extracted from the tabular examples.
    Accepts either a dict with keys 'x0'..'x7' or a sequence of length>=8.
    Returns 0/1.
    """
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (x['x0'],x['x1'],x['x2'],x['x3'],x['x4'],x['x5'],x['x6'],x['x7'])
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = x[:8]

    # Main "positive" cluster: high x2 and strongly negative x3 with mild x6.
    if (x6 > -4.5 and x3 < -1.0) or (x2 > 3.0):
        return 1

    # Secondary positives: moderately high x2 with negative x3, not-too-large |x6|, and sizable x4.
    if x2 > 2.5 and x3 < -0.5 and x6 > -6.0 and x4 > 10.0:
        return 1

    # Rare mid-x2 positives when x7 is not extremely negative.
    if x7 > -60.0 and x1 <= -26.0 and x2 > 2.0 and x6 > -6.5:
        return 1
    if x7 > -60.0 and x4 < 6.0 and x1 < -24.5 and x2 > 2.0 and x6 > -7.0:
        return 1

    return 0
h_1 = f

ALPHAS = [1.464056042939506]
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
