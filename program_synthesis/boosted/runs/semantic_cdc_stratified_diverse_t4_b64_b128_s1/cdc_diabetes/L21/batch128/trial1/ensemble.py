"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    # simple risk-score rule (non-trainable) for diabetes indicator
    sev = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    def s(name):
        return sev.get(x.get(name, "medium"), 2)

    def yes(name):
        return x.get(name) == "yes"

    score = 0

    # cardiometabolic factors
    if yes("HighBP"): score += 2
    if yes("HighChol"): score += 2

    bmi = s("BMI")
    if bmi >= 3: score += 1
    if bmi >= 4: score += 1

    age = s("Age")
    if age >= 3: score += 1
    if age >= 4: score += 1

    # perceived/functional health
    gh = s("GenHlth")
    if gh >= 3: score += 1
    if gh >= 4: score += 1

    if yes("DiffWalk"): score += 2

    # major vascular disease history
    if yes("Stroke"): score += 2
    if yes("HeartDiseaseorAttack"): score += 2

    # lifestyle
    if x.get("PhysActivity") == "no": score += 1

    # strong shortcut patterns
    if (yes("Stroke") or yes("HeartDiseaseorAttack")) and (yes("HighBP") or yes("HighChol")):
        return 1

    # require at least one core metabolic risk to call positive
    core = (yes("HighBP") or yes("HighChol") or bmi >= 3)

    return int(core and score >= 6)
h_1 = f

ALPHAS = [0.3965579899706428]
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
