"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic boolean rule for 8 features (x0..x7). Accepts list/tuple or dict."""
    if isinstance(x, dict):
        x0,x1,x2,x3,x4,x5,x6,x7 = (x[f"x{i}"] for i in range(8))
    else:
        x0,x1,x2,x3,x4,x5,x6,x7 = x

    # Regime 1: very small x7 with large x5 (often the strongest signal)
    c1 = (x7 < 3.2) and (x5 > 15.0) and (x0 < 33.0) and (x3 < -0.5)

    # Regime 2: low x0 with clearly negative x3 (but not the high-x2 stable cases)
    c2 = (x0 < 27.0) and (x3 < -2.5) and (x2 < 1.30) and (x7 < 10.0)

    # Regime 3: moderate x7, but relatively small x6 with elevated x5 and negative x3
    c3 = (x0 < 32.0) and (x6 < 9.5) and (x5 > 12.0) and (x3 < -1.0)

    # Regime 4: large x5 with strongly negative x4, provided x3 is not near 0
    c4 = (x0 < 35.0) and (x4 < -5.0) and (x5 > 25.0) and (x3 < -1.0)

    return int(c1 or c2 or c3 or c4)
h_1 = f

ALPHAS = [1.182139330999693]
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
