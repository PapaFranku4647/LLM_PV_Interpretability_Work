"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    # Heuristic risk-score rule for diabetes (binary), using discretized CDC indicators.
    vmap = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    def cat(name, default=2):
        return vmap.get(str(x.get(name, "")).strip().lower(), default)

    def yes(name):
        return str(x.get(name, "no")).strip().lower() == "yes"

    bmi = cat("BMI")
    age = cat("Age")
    gen = cat("GenHlth")  # higher assumed worse health

    # Strong clinical-event overrides
    if yes("Stroke") and (yes("HighBP") or age >= 3 or yes("HeartDiseaseorAttack")):
        return 1
    if yes("HeartDiseaseorAttack") and (age >= 3 or bmi >= 2 or yes("HighBP")):
        return 1

    score = 0
    score += 2 if yes("HighBP") else 0
    score += 1 if yes("HighChol") else 0

    score += 2 if bmi >= 3 else (1 if bmi == 2 else 0)
    score += 2 if age >= 3 else (1 if age == 2 else 0)
    score += 2 if gen >= 3 else (1 if gen == 2 else 0)

    score += 2 if yes("DiffWalk") else 0
    score += 2 if yes("Stroke") else 0
    score += 2 if yes("HeartDiseaseorAttack") else 0

    score += 1 if yes("Smoker") else 0
    score -= 1 if yes("PhysActivity") else 0

    # Metabolic syndrome style bump
    if yes("HighBP") and yes("HighChol") and bmi >= 2:
        score += 1

    return 1 if score >= 6 else 0
h_1 = f

ALPHAS = [0.427225483354969]
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
