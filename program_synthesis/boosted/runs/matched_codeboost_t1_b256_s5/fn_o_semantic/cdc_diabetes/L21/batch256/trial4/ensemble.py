"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    yes = lambda k: x.get(k) == "yes"

    age = x.get("Age")
    bmi = x.get("BMI")
    gen = x.get("GenHlth")
    phys = x.get("PhysHlth")
    inc = x.get("Income")

    score = 0

    # Core metabolic / cardio risks
    if yes("HighBP"):
        score += 3
    if yes("HighChol"):
        score += 2
    if bmi == "high":
        score += 2
    elif bmi == "very high":
        score += 3

    # Age
    if age == "medium":
        score += 1
    elif age == "high":
        score += 2
    elif age == "very high":
        score += 3

    # Functional / comorbidity indicators
    if yes("DiffWalk"):
        score += 2
    if yes("HeartDiseaseorAttack"):
        score += 2
    if yes("Stroke"):
        score += 2

    # Self-reported health (assume higher bin = worse)
    if gen in {"high", "very high"}:
        score += 2
    if phys in {"high", "very high"}:
        score += 1

    # Lifestyle / SES (weak signals)
    if x.get("PhysActivity") == "no":
        score += 1
    if yes("Smoker"):
        score += 1
    if inc in {"very low", "low"}:
        score += 1

    # Decision rule: without HighBP, require stronger evidence
    if yes("HighBP"):
        return 1 if score >= 7 else 0
    else:
        strong = (age in {"high", "very high"} or bmi == "very high" or yes("HeartDiseaseorAttack")
                  or yes("DiffWalk") or gen in {"high", "very high"})
        return 1 if (strong and score >= 9) else 0
h_1 = f

ALPHAS = [0.4788379740185047]
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
