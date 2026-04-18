"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Rule-based binary classifier.

    Accepts x as either:
      - dict with keys 'x0'..'x7', or
      - list/tuple with 8 values in order.
    Returns 0/1.
    """
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (float(x[f"x{i}"]) for i in range(8))
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = map(float, x[:8])

    # Clear positive region: high x2 (often paired with strongly negative x3)
    if x2 >= 3.0:
        return 1

    # Large x4 with near-zero x7 tends to be positive unless x0 is very large
    if (x0 < 210) and (x2 > 1.8) and (x4 > 25) and (x7 > -10):
        return 1

    # Medium x2 with sufficiently negative x3 and non-trivial x5
    if (x2 > 2.6) and (x3 < -0.5) and (x5 > 15):
        return 1

    # Another positive pocket: higher x5 with modest x3 and not-too-negative x7
    if (x2 > 2.2) and (x5 > 20) and (x3 < 0.8) and (x7 > -10):
        return 1

    # Broad catch for moderate conditions (kept conservative via x7/x0 bounds)
    if (x0 < 210) and (x2 > 1.8) and (x5 > 16) and (x3 < 1.6) and (x7 > -40):
        return 1

    return 0
h_1 = f

ALPHAS = [1.424940198270714]
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
