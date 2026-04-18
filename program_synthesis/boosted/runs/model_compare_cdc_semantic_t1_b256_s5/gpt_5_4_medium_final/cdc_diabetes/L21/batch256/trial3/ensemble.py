"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    o={'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    bmi=o.get(x.get('BMI'),2)
    age=o.get(x.get('Age'),2)
    gen=o.get(x.get('GenHlth'),2)
    phys=o.get(x.get('PhysHlth'),2)
    s=0
    s+=2*(x.get('HighBP')=='yes')
    s+=1*(x.get('HighChol')=='yes')
    s+=3*(x.get('HeartDiseaseorAttack')=='yes')
    s+=2*(x.get('Stroke')=='yes')
    s+=2*(x.get('DiffWalk')=='yes')
    s+=1*(x.get('PhysActivity')=='no')
    s+=max(0,bmi-1)
    s+=max(0,age-1)
    s+=max(0,gen-2)
    s+=max(0,phys-2)
    s+=1*(x.get('Income') in ('very low','low'))
    s+=1*(x.get('Education') in ('very low','low'))
    s-=1*(x.get('HvyAlcoholConsump')=='yes')
    return int(s>=6)
h_1 = f

ALPHAS = [0.4788379740185047]
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
