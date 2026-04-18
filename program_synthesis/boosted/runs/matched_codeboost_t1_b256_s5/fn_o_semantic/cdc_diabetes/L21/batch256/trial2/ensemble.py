"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule/score for diabetes risk from discretized CDC indicators."""
    lvl_map = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    def yn(key):
        return (x.get(key) == "yes")

    def lvl(key, default=2):
        v = x.get(key)
        return lvl_map.get(v, default)

    points = 0

    # Major medical history
    if yn("Stroke"):
        points += 5
    if yn("HeartDiseaseorAttack"):
        points += 5

    # Core metabolic risk
    if yn("HighBP"):
        points += 3
    if yn("HighChol"):
        points += 2
    if lvl("BMI") >= 3:  # high/very high
        points += 3

    # Function/health status
    if yn("DiffWalk"):
        points += 2
    points += max(0, lvl("GenHlth") - 2)  # medium->0, high->1, very high->2

    # Age
    points += max(0, lvl("Age") - 2)      # medium->0, high->1, very high->2

    # Lifestyle / SES (small effects)
    if yn("Smoker"):
        points += 1
    if x.get("PhysActivity") == "no":
        points += 1
    if lvl("Income") <= 1:
        points += 1
    if lvl("Education") == 0:
        points += 1

    return 1 if points >= 7 else 0
h_1 = f

ALPHAS = [0.4985010561286636]
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
