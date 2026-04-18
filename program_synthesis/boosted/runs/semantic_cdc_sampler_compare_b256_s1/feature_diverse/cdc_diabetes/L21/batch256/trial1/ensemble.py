"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    lvl = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    yn = lambda k: x.get(k) == "yes"

    bmi = lvl.get(x.get("BMI", ""), 2)
    age = lvl.get(x.get("Age", ""), 2)
    gen = lvl.get(x.get("GenHlth", ""), 2)
    inc = lvl.get(x.get("Income", ""), 2)

    # Strong vascular comorbidity patterns
    if yn("Stroke") and (yn("HighBP") or yn("HighChol") or bmi >= 3 or age >= 2 or yn("DiffWalk")):
        return 1
    if yn("HeartDiseaseorAttack") and (yn("HighBP") or yn("HighChol") or age >= 2 or yn("DiffWalk")):
        return 1

    # Simple additive risk score
    risk = 0
    risk += 3 if yn("HighBP") else 0
    risk += 2 if yn("HighChol") else 0
    risk += 2 if bmi >= 3 else (1 if bmi == 2 else 0)
    risk += 2 if age >= 3 else (1 if age == 2 else 0)
    risk += 2 if yn("DiffWalk") else 0
    risk += 1 if yn("Smoker") else 0
    risk += 1 if x.get("PhysActivity") == "no" else 0
    risk += 1 if gen >= 3 else 0
    risk += 1 if inc <= 1 else 0

    return 1 if risk >= 8 else 0
h_1 = f

ALPHAS = [0.4198451841958359]
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
