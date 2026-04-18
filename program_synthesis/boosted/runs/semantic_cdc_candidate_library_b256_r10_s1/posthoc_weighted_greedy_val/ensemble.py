"""Auto-generated post-hoc boosted ensemble wrapper."""
from typing import Any

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

# Learner 1: row 8, attempt 9
def f(x):
    # Rule-based risk score for diabetes (0/1) using key CDC indicators.
    yn = lambda k: (x.get(k, "no") == "yes")

    ord5 = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    bmi = x.get("BMI", "medium")
    age = x.get("Age", "medium")
    gen = x.get("GenHlth", "medium")
    inc = x.get("Income", "medium")
    edu = x.get("Education", "medium")

    score = 0.0

    # Strong clinical correlates
    score += 2.0 * yn("HighBP")
    score += 1.5 * yn("HighChol")
    score += 2.0 * yn("Stroke")
    score += 2.0 * yn("HeartDiseaseorAttack")
    score += 1.5 * yn("DiffWalk")

    # Anthropometrics and age
    score += {"very low": -0.5, "low": 0.0, "medium": 0.5, "high": 1.5, "very high": 2.5}.get(bmi, 0.5)
    score += {"very low": 0.0, "low": 0.5, "medium": 1.0, "high": 1.7, "very high": 2.3}.get(age, 1.0)

    # Self-rated health (worse -> higher risk)
    score += 0.6 * ord5.get(gen, 2)

    # Lifestyle and access (smaller effects)
    score += 0.5 * yn("Smoker")
    score -= 1.0 * yn("PhysActivity")
    score -= 0.3 * yn("HvyAlcoholConsump")

    # Socioeconomic gradient (lower -> higher risk)
    score += 0.25 * (2 - min(2, ord5.get(inc, 2)))
    score += 0.15 * (2 - min(2, ord5.get(edu, 2)))

    # Require at least one major metabolic/cardiovascular flag unless score is very high
    major_flag = yn("HighBP") or yn("HighChol") or (bmi in ("high", "very high"))

    return int((score >= 6.0 and major_flag) or (score >= 7.5))
h_1 = f

ALPHAS = [0.43105187371774]
DIRECTIONS = [1]
LEARNERS = [h_1]

def f(x: Any) -> int:
    score = 0.0
    for alpha, direction, learner in zip(ALPHAS, DIRECTIONS, LEARNERS):
        score += alpha * direction * _normalize_pred_to_pm1(learner(x))
    return 1 if score >= 0.0 else 0
