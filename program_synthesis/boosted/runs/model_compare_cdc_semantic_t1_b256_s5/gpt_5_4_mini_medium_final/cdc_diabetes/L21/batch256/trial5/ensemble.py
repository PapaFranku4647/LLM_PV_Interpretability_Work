"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
 b=x.get('BMI'); a=x.get('Age'); g=x.get('GenHlth'); p=x.get('PhysHlth')
 return int(x.get('Stroke')=='yes' or x.get('HeartDiseaseorAttack')=='yes' or (b=='very high' and (x.get('HighBP')=='yes' or x.get('HighChol')=='yes' or a in {'medium','high','very high'} or x.get('DiffWalk')=='yes' or g in {'high','very high'})) or (b=='high' and x.get('HighBP')=='yes' and (x.get('HighChol')=='yes' or a in {'high','very high'} or x.get('DiffWalk')=='yes')) or (x.get('HighBP')=='yes' and x.get('HighChol')=='yes' and b in {'medium','high','very high'}) or (b=='medium' and x.get('HighBP')=='yes' and (x.get('HighChol')=='yes' or a in {'high','very high'} or g in {'high','very high'} or p in {'high','very high'} or x.get('DiffWalk')=='yes')))
h_1 = f

ALPHAS = [0.4886257158319211]
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
