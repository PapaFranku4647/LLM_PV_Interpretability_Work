"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    hi = {'high', 'very high'}
    score = 0
    if x.get('HighBP') == 'yes':
        score += 2
    if x.get('HighChol') == 'yes':
        score += 1
    if x.get('BMI') == 'very high':
        score += 2
    elif x.get('BMI') == 'high' and (x.get('Age') in hi or x.get('PhysActivity') == 'no' or x.get('GenHlth') in hi):
        score += 1
    if x.get('Age') in hi:
        score += 1
    if x.get('GenHlth') == 'very high':
        score += 2
    elif x.get('GenHlth') == 'high':
        score += 1
    if x.get('DiffWalk') == 'yes':
        score += 1
    if x.get('PhysActivity') == 'no':
        score += 1
    if x.get('HeartDiseaseorAttack') == 'yes':
        score += 1
    if x.get('Stroke') == 'yes':
        score += 1
    if x.get('Age') in hi and x.get('DiffWalk') == 'yes':
        score += 2
    if x.get('HighBP') == 'yes' and x.get('BMI') == 'very high':
        score += 1
    return int(score >= 6)
h_1 = f

ALPHAS = [0.4499707969363128]
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
