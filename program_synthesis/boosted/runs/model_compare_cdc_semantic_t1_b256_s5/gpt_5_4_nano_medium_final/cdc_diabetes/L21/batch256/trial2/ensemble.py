"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def yn(k, yes_value="yes"):
        return 1 if x.get(k) == yes_value else 0

    def ord5(k):
        m = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
        return m.get(x.get(k), 0)

    risk = 0.0

    # Cardiometabolic conditions
    risk += 2.0 * yn("HighBP")
    risk += 2.0 * yn("HighChol")
    risk += 3.0 * yn("Stroke")
    risk += 2.0 * yn("HeartDiseaseorAttack")
    risk += 1.0 * yn("Smoker")

    # Functional limitation / perceived health
    risk += 2.0 * yn("DiffWalk")
    risk += 1.5 * ord5("GenHlth")

    # Body/age
    risk += 1.0 * ord5("BMI")
    risk += 0.8 * ord5("Age")

    # Activity and health impacts
    risk += 1.5 * (1 if x.get("PhysActivity") == "no" else 0)
    risk += 0.6 * ord5("PhysHlth")
    risk += 0.3 * ord5("MentHlth")

    # Lifestyle (weaker signals)
    risk += 0.4 * (1 if x.get("Fruits") == "no" else 0)
    risk += 0.4 * (1 if x.get("Veggies") == "no" else 0)
    risk += 0.4 * yn("HvyAlcoholConsump")

    # Access to care
    risk += 1.0 * (1 if x.get("AnyHealthcare") == "no" else 0)
    risk += 1.0 * yn("NoDocbcCost")

    return 1 if risk >= 11.5 else 0
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
