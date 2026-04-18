"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic, non-trainable rule/score for CDC diabetes indicators (binary output)."""
    def yn(k):
        return 1 if str(x.get(k, "no")).strip().lower() == "yes" else 0

    def ord5(k):
        v = str(x.get(k, "medium")).strip().lower()
        m = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
        return m.get(v, 2)

    bmi = str(x.get("BMI", "medium")).strip().lower()
    age = str(x.get("Age", "medium")).strip().lower()

    highbp = yn("HighBP")
    highchol = yn("HighChol")
    heart = yn("HeartDiseaseorAttack")
    stroke = yn("Stroke")
    diffwalk = yn("DiffWalk")
    smoker = yn("Smoker")
    physact = yn("PhysActivity")

    # High-precision rules (strong correlates)
    if highbp and highchol and (bmi in {"high", "very high"} or age in {"high", "very high"} or heart or diffwalk):
        return 1
    if bmi == "very high" and (highbp or highchol) and (age in {"medium", "high", "very high"} or diffwalk):
        return 1

    # Risk score fallback
    score = 0
    score += 3 * highbp
    score += 3 * highchol
    score += 3 * heart
    score += 2 * stroke
    score += 2 * diffwalk

    # BMI/Age (ordinal)
    score += {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}.get(bmi, 2)
    score += {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}.get(age, 2)

    # General/physical health (higher = worse)
    gh = ord5("GenHlth")
    ph = ord5("PhysHlth")
    score += max(0, gh - 1)  # medium+ adds risk
    score += max(0, ph - 1)

    score += 1 * smoker
    score -= 1 * physact

    # Socioeconomic (lower = higher risk)
    inc = str(x.get("Income", "medium")).strip().lower()
    edu = str(x.get("Education", "medium")).strip().lower()
    score += 2 if inc == "very low" else (1 if inc == "low" else 0)
    score += 1 if edu in {"very low", "low"} else 0

    # Access to care (weak)
    score += 1 if str(x.get("AnyHealthcare", "yes")).strip().lower() == "no" else 0

    return 1 if score >= 12 else 0
h_1 = f

ALPHAS = [0.4339294874465056]
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
