"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def yes(key):
        v = x.get(key)
        return 1 if v == "yes" else 0
    def no(key):
        v = x.get(key)
        return 1 if v == "no" else 0
    def ordv(key):
        m = {"very low": 1, "low": 2, "medium": 3, "high": 4, "very high": 5}
        v = x.get(key)
        return m.get(v, 3)

    score = 0.0
    # Clinical risk factors
    score += 2.0 * yes("HighBP")
    score += 1.7 * yes("HighChol")
    score += 1.2 * yes("Smoker")
    score += 2.0 * yes("Stroke")
    score += 1.5 * yes("HeartDiseaseorAttack")
    score += 1.4 * yes("DiffWalk")

    # Health status / demographics (ordinal bins)
    score += 2.0 * (ordv("BMI") - 1) / 4.0
    score += 1.6 * (ordv("Age") - 1) / 4.0
    score += 2.0 * (ordv("GenHlth") - 1) / 4.0

    # Lifestyle / access
    score += 1.0 * no("PhysActivity")
    score += 0.6 * no("AnyHealthcare")
    score += 0.6 * yes("NoDocbcCost")
    score += 0.4 * no("CholCheck")
    score += 0.3 * yes("HvyAlcoholConsump")
    score += 0.2 * (x.get("Sex") == "male")

    # Slight protection
    score -= 0.3 * yes("Fruits")
    score -= 0.3 * yes("Veggies")

    # Symptoms/impairment severity
    score += 0.8 * (ordv("PhysHlth") - 1) / 4.0
    score += 0.6 * (ordv("MentHlth") - 1) / 4.0

    return 1 if score >= 5.8 else 0
h_1 = f

ALPHAS = [0.4405044588647577]
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
