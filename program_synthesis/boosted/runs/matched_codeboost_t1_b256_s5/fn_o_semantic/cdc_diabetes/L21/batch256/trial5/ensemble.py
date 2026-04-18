"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    ordmap = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
    yn = lambda k: 1 if x.get(k) == 'yes' else 0
    v = lambda k, d='medium': ordmap.get(x.get(k, d), ordmap[d])

    score = 0
    # cardio/metabolic
    score += 2 * yn('HighBP')
    score += 2 * yn('HighChol')
    score += max(0, v('BMI') - 1)          # medium/high/very high raise risk
    score += v('Age')                       # older => higher risk

    # poor overall health / disability
    score += max(0, v('GenHlth') - 1)
    score += 2 * yn('DiffWalk')

    # major comorbidities
    score += 3 * yn('Stroke')
    score += 2 * yn('HeartDiseaseorAttack')

    # lifestyle
    score += 1 * yn('Smoker')
    score -= 1 * yn('PhysActivity')
    score -= 1 * yn('Fruits')
    score -= 1 * yn('Veggies')

    # socioeconomic (weak signals)
    score += 1 if v('Income') <= 1 else 0
    score += 1 if v('Education') <= 1 else 0

    return 1 if score >= 7 else 0
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
