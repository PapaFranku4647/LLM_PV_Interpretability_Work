"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    s = {
        'very low': -1, 'low': 0, 'medium': 0.5, 'high': 1.5, 'very high': 2
    }.get(x.get('BMI'), 0)
    s += {
        'very low': 0, 'low': 0.5, 'medium': 0, 'high': 1.5, 'very high': 2.5
    }.get(x.get('Age'), 0)
    s += {
        'very low': 0, 'low': 0, 'medium': 0, 'high': 1, 'very high': 1.5
    }.get(x.get('GenHlth'), 0)
    s += 2.5 if x.get('HighBP') == 'yes' else 0
    s += 1 if x.get('HighChol') == 'yes' else 0
    s += 1.5 if x.get('DiffWalk') == 'yes' else 0
    s += 1 if x.get('PhysActivity') == 'no' else 0
    s += 1.5 if x.get('Stroke') == 'yes' else 0
    s += 0.5 if x.get('Smoker') == 'yes' else 0
    if x.get('HeartDiseaseorAttack') == 'yes' and (x.get('DiffWalk') == 'yes' or x.get('GenHlth') in {'high', 'very high'} or x.get('Age') in {'high', 'very high'}):
        s += 1
    if x.get('Age') == 'very high' and x.get('PhysHlth') in {'high', 'very high'}:
        s += 1.5
    if x.get('Education') in {'very low', 'low'}:
        s += 0.5
    if x.get('Income') in {'very low', 'low'}:
        s += 0.5
    if x.get('HighBP') == 'yes' and x.get('BMI') in {'high', 'very high'} and (x.get('HighChol') == 'yes' or x.get('Smoker') == 'yes' or x.get('PhysActivity') == 'no' or x.get('Age') != 'very low'):
        return 1
    return int(s >= 6)
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
