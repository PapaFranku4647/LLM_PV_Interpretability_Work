"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    a,b,g=x.get('Age'),x.get('BMI'),x.get('GenHlth')
    s,d,h=x.get('Smoker'),x.get('DiffWalk'),x.get('HeartDiseaseorAttack')
    if h=='yes' or x.get('Stroke')=='yes': return 1
    if x.get('HighBP')=='yes': return int((b in {'medium','high','very high'} and (a in {'high','very high'} or g in {'medium','high','very high'} or d=='yes' or h=='yes')) or (a in {'high','very high'} and s=='yes' and b in {'low','medium'}))
    if x.get('HighChol')=='yes': return int((a in {'high','very high'} and (s=='yes' or b in {'high','very high'})) or (b in {'high','very high'} and a in {'medium','high','very high'}))
    return int((a=='very low' and b in {'medium','high','very high'}) or (a in {'high','very high'} and s=='yes' and b in {'low','medium'}))
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
