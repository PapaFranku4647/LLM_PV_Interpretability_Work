"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    s=0
    if x.get('HighBP')=='yes': s+=2
    if x.get('HighChol')=='yes': s+=1
    if x.get('HeartDiseaseorAttack')=='yes': s+=2
    if x.get('Stroke')=='yes': s+=2
    if x.get('DiffWalk')=='yes': s+=1
    if x.get('PhysActivity')=='no': s+=1
    if x.get('GenHlth') in {'high','very high'}: s+=1
    if x.get('Age')=='medium': s+=1
    if x.get('Age') in {'high','very high'}: s+=2
    s += {'very low':-2,'low':-1,'medium':1,'high':2,'very high':3}.get(x.get('BMI'),0)
    return int(s>=4)
h_1 = f

ALPHAS = [0.4217901919935356]
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
