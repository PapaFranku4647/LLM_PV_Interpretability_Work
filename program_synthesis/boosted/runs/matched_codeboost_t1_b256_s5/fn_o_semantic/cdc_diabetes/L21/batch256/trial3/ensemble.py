"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def lvl(v):
        m = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
        return m.get((v or "").strip().lower(), 2)

    def yes(k):
        return (x.get(k, "").strip().lower() == "yes")

    bmi = (x.get("BMI") or "").strip().lower()
    age = (x.get("Age") or "").strip().lower()
    gen = (x.get("GenHlth") or "").strip().lower()

    # Strong low-risk shortcut
    if (not yes("HighBP") and not yes("HighChol") and bmi in {"very low", "low"} and lvl(age) <= 1 and lvl(gen) <= 1):
        return 0

    score = 0.0

    # Core metabolic / vascular risk
    if yes("HighBP"): score += 2.2
    if yes("HighChol"): score += 1.6

    if bmi in {"very high"}: score += 2.4
    elif bmi in {"high"}: score += 2.0
    elif bmi in {"medium"}: score += 1.0

    a = lvl(age)
    if a >= 4: score += 2.4
    elif a == 3: score += 1.8
    elif a == 2: score += 0.8

    # Functional / comorbidity signals
    if yes("DiffWalk"): score += 2.0
    if yes("HeartDiseaseorAttack"): score += 1.4
    if yes("Stroke"): score += 1.0

    # Self-reported general health (worse -> higher risk)
    score += max(0, lvl(gen) - 1) * 0.9

    # Smaller correlates
    if yes("Smoker"): score += 0.3
    if (x.get("Income", "").strip().lower() in {"very low", "low"}): score += 0.4
    if (x.get("Education", "").strip().lower() in {"very low", "low"}): score += 0.2

    # Protective activity
    if yes("PhysActivity"): score -= 1.0

    # High-risk override pattern
    if yes("HighBP") and bmi in {"high", "very high"} and lvl(age) >= 2:
        score += 1.0

    return int(score >= 6.0)
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
