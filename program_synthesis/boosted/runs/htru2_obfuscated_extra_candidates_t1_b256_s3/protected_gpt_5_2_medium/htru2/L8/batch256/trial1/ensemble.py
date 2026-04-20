"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule-based classifier for 8 features x0..x7.
    Accepts a list/tuple (len>=8) or a dict with keys 'x0'..'x7'."""
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (float(x[f'x{i}']) for i in range(8))
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = (float(x[i]) for i in range(8))

    # Very-small x0 regime is almost always positive
    if x0 <= 22.5:
        return 1

    # Extremely negative x3 indicates positive
    if x3 <= -20.0:
        return 1

    # Small-x7/low-x6 regime with sufficiently negative x3
    if (x7 <= 3.0 and x3 <= -4.0):
        return 1

    # Strongly negative x4 with large x5 and small x6
    if (x4 <= -10.0 and x5 >= 25.0 and x6 <= 3.5):
        return 1

    # Moderate x3 negativity but large x6 (seen in some mid-range positives)
    if (x3 <= -3.0 and x6 >= 12.0 and x5 >= 6.0):
        return 1

    # Mid-range positives with modest x6 and more negative x3
    if (x0 <= 33.0 and x3 <= -2.3 and x6 <= 9.0 and x5 >= 10.0):
        return 1

    # Low x1 with sufficiently negative x3 in mid/high x0
    if (x0 <= 34.0 and x1 <= 105.0 and x3 <= -2.4):
        return 1

    return 0
h_1 = f

ALPHAS = [0.8581678537054065]
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
