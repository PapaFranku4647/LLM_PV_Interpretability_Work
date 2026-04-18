"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    # Ordinal bins (assumed increasing risk): very low < low < medium < high < very high
    ORD = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    def o(key, default="medium"):
        return ORD.get(x.get(key, default), ORD[default])

    risk = 0.0

    # Core cardio-metabolic risks
    if x.get("HighBP") == "yes":
        risk += 3.0
    if x.get("HighChol") == "yes":
        risk += 2.0

    bmi = o("BMI")
    risk += {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0}.get(bmi, 1.0)

    age = o("Age")
    risk += {0: 0.0, 1: 0.3, 2: 0.8, 3: 1.5, 4: 2.2}.get(age, 0.8)

    # Vascular events / comorbidities
    if x.get("Stroke") == "yes":
        risk += 2.5
    if x.get("HeartDiseaseorAttack") == "yes":
        risk += 2.5
    if x.get("DiffWalk") == "yes":
        risk += 1.2

    # Health status / behavior
    if x.get("PhysActivity") == "no":
        risk += 1.0
    if x.get("Smoker") == "yes":
        risk += 0.6

    gen = o("GenHlth")
    if gen >= 3:
        risk += 0.8 + 0.6 * (gen - 3)  # high/very high worse general health

    ph = o("PhysHlth")
    if ph >= 3:
        risk += 0.4 + 0.3 * (ph - 3)

    # Socioeconomic protection (weak)
    inc = o("Income")
    edu = o("Education")
    if inc >= 3:
        risk -= 0.5
    if edu >= 3:
        risk -= 0.3

    # Gate: require at least one major metabolic/cardiac driver
    major = (
        x.get("HighBP") == "yes" or
        x.get("HighChol") == "yes" or
        bmi >= 3 or
        age >= 3 or
        x.get("Stroke") == "yes" or
        x.get("HeartDiseaseorAttack") == "yes"
    )

    return int(major and risk >= 7.0)
h_1 = f

ALPHAS = [0.4303335217347083]
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
