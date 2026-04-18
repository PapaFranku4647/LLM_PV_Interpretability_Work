"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule-based diabetes indicator from discretized CDC features."""
    def yn(k):
        return (x.get(k) or "").strip().lower()

    def ordv(k):
        m = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
        return m.get((x.get(k) or "").strip().lower(), 2)

    highbp = yn("HighBP") == "yes"
    highchol = yn("HighChol") == "yes"
    smoker = yn("Smoker") == "yes"
    stroke = yn("Stroke") == "yes"
    heart = yn("HeartDiseaseorAttack") == "yes"
    diffwalk = yn("DiffWalk") == "yes"
    inactive = yn("PhysActivity") == "no"

    bmi = ordv("BMI")
    age = ordv("Age")
    gen = ordv("GenHlth")  # higher ~ worse
    inc = ordv("Income")   # higher ~ better

    # Strong-signal shortcuts
    if (stroke or heart) and (highbp or bmi >= 3 or diffwalk):
        return 1

    score = 0
    score += 2 if highbp else 0
    score += 1 if highchol else 0
    score += 1 if bmi >= 3 else 0
    score += 1 if bmi == 4 else 0
    score += 1 if age >= 3 else 0
    score += 1 if gen >= 3 else 0
    score += 1 if diffwalk else 0
    score += 1 if inactive else 0
    score += 1 if smoker else 0
    score += 2 if stroke else 0
    score += 2 if heart else 0
    score += 1 if inc <= 1 else 0

    # Decision boundary
    if score >= 6:
        return 1
    if score >= 5 and (age >= 2 or bmi >= 3 or highbp):
        return 1
    if score >= 4 and highbp and (highchol or bmi >= 3) and age >= 2:
        return 1
    return 0
h_1 = f

ALPHAS = [0.4691348192964651]
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
