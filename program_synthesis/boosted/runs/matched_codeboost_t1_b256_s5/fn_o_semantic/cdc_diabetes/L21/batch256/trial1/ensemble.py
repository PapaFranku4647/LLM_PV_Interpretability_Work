"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic, non-trainable rule to approximate diabetes risk from CDC-style binned features."""
    order = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    def yn(k):
        return x.get(k, "no") == "yes"

    def v(k, default="medium"):
        return order.get(x.get(k, default), order[default])

    highbp = yn("HighBP")
    highchol = yn("HighChol")
    stroke = yn("Stroke")
    heart = yn("HeartDiseaseorAttack")
    diffwalk = yn("DiffWalk")

    bmi = v("BMI")
    age = v("Age")
    gen = v("GenHlth")

    # Strong comorbidity rule
    if (stroke or heart) and (highbp or highchol or bmi >= 2) and (age >= 1 or gen >= 2 or diffwalk):
        return 1

    # Guard: very low BMI rarely predicts diabetes unless other strong signals exist
    if bmi == 0 and not (highbp and highchol and (diffwalk or gen >= 3 or stroke or heart)):
        return 0

    score = 0.0
    score += 2.0 if highbp else 0.0
    score += 1.5 if highchol else 0.0
    score += 1.0 if diffwalk else 0.0
    score += 2.0 if heart else 0.0
    score += 2.0 if stroke else 0.0

    # BMI / age (binned)
    score += (-0.5 if bmi == 0 else 0.0)
    score += 0.8 * max(0, bmi - 1)   # medium/high/very high increases risk
    score += 0.7 * max(0, age - 1)   # older age increases risk

    # General health (very high ~ worst)
    score += 1.0 if gen >= 3 else (0.5 if gen == 2 else 0.0)

    # Lifestyle modifiers
    score += 0.5 if yn("Smoker") else 0.0
    score += (0.5 if x.get("PhysActivity", "yes") == "no" else -0.5)
    score += (-0.4 if yn("HvyAlcoholConsump") else 0.0)

    # Socioeconomic weak signal
    score += 0.4 if v("Income") <= 1 else 0.0

    # Extra guard: very young + modest score tends to be negative
    if age <= 1 and not (stroke or heart) and score < 5.5:
        return 0

    return 1 if score >= 4.0 else 0
h_1 = f

ALPHAS = [0.4595133558465149]
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
