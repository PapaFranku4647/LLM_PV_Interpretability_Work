"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x: dict) -> int:
    def bval(v, order=("very low","low","medium","high","very high")):
        if v is None:
            return 0.0
        try:
            return float(order.index(v)) / (len(order) - 1)  # 0..1
        except ValueError:
            return 0.0

    def yn(key, yes_true=True, w_yes=1.0, w_no=0.0):
        v = x.get(key)
        if v is None:
            return 0.0
        if v == "yes":
            return w_yes if yes_true else w_no
        if v == "no":
            return w_no if yes_true else w_yes
        return 0.0

    score = 0.0

    # Major medical risk factors
    score += yn("HighBP", w_yes=2.0, w_no=0.0)
    score += yn("HighChol", w_yes=1.6, w_no=0.0)
    score += yn("Stroke", w_yes=3.0, w_no=0.0)
    score += yn("HeartDiseaseorAttack", w_yes=2.4, w_no=0.0)

    # Lifestyle / functional status
    score += yn("Smoker", w_yes=1.0, w_no=0.0)
    score += yn("PhysActivity", yes_true=False, w_yes=1.6, w_no=0.0)  # no activity
    score += yn("DiffWalk", w_yes=2.0, w_no=0.0)

    # Protective behaviors
    score += yn("Fruits", yes_true=False, w_yes=0.6, w_no=0.0)
    score += yn("Veggies", yes_true=False, w_yes=0.6, w_no=0.0)

    # Alcohol / access (weaker signals)
    score += yn("HvyAlcoholConsump", w_yes=0.5, w_no=0.0)
    score += yn("NoDocbcCost", w_yes=0.3, w_no=0.0)
    score += yn("AnyHealthcare", yes_true=False, w_yes=0.8, w_no=0.0)  # no healthcare

    # Ordinal severity features (assume "very high" is worse)
    score += 2.2 * bval(x.get("BMI"))
    score += 2.6 * bval(x.get("Age"))
    score += 2.1 * bval(x.get("GenHlth"))
    score += 0.9 * bval(x.get("MentHlth"))
    score += 1.0 * bval(x.get("PhysHlth"))

    # Socioeconomic (lower is often worse)
    # Here we treat low/very low as higher risk by reversing the scale.
    inc = bval(x.get("Income"))
    edu = bval(x.get("Education"))
    score += 1.3 * (1.0 - inc) + 0.9 * (1.0 - edu)

    return 1 if score >= 8.0 else 0
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
