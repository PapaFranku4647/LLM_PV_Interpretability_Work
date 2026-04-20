"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    r={'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    y=lambda k: x.get(k,'no')=='yes'
    g=lambda k: r.get(x.get(k,''),0)
    bp,chol,bmi,age=g('HighBP')==1,y('HighChol'),g('BMI'),g('Age')
    gh,ph,diff=g('GenHlth'),g('PhysHlth'),y('DiffWalk')
    heart,stroke=y('HeartDiseaseorAttack'),y('Stroke')
    if heart or stroke:
        return int(bp or bmi>=2 or age>=3 or gh>=3)
    if bmi==4:
        return int(bp or chol or age>=3 or gh>=3 or ph>=3)
    if bmi==3:
        return int((bp and (age>=2 or chol or diff or gh>=3)) or (chol and age>=3))
    if bp and bmi>=2 and (age>=2 or chol or diff or gh>=3 or ph>=3):
        return 1
    if bp and age==4 and bmi<=1 and not chol:
        return 1
    if chol and bmi>=3 and age>=3 and (bp or gh>=3):
        return 1
    if age==4 and bmi>=3 and (bp or chol):
        return 1
    return 0
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
