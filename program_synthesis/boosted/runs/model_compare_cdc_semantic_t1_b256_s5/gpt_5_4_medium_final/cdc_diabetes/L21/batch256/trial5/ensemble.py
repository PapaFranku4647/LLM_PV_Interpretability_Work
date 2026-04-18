"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    r={"very low":0,"low":1,"medium":2,"high":3,"very high":4}
    age=r.get(x.get("Age"),0); bmi=r.get(x.get("BMI"),0); gh=r.get(x.get("GenHlth"),0); ph=r.get(x.get("PhysHlth"),0)
    if (x.get("HighBP")=="no" and x.get("HighChol")=="no" and x.get("HeartDiseaseorAttack")=="no" and x.get("Stroke")=="no" and x.get("DiffWalk")=="no" and x.get("PhysActivity")=="yes" and gh<=1 and bmi<=2 and age<=1):
        return 0
    score=0
    score+=2*(x.get("HighBP")=="yes")
    score+=1*(x.get("HighChol")=="yes")
    score+=2*(bmi>=4)+1*(bmi==3)
    score+=2*(age>=3)+1*(age==2)
    score+=2*(gh>=3)+1*(gh==2)
    score+=1*(ph>=3)
    score+=2*(x.get("DiffWalk")=="yes")
    score+=2*(x.get("HeartDiseaseorAttack")=="yes")
    score+=1*(x.get("Stroke")=="yes")
    score+=1*(x.get("PhysActivity")=="no")
    score+=1*(x.get("Education") in {"very low","low"})
    score+=1*(x.get("Income") in {"very low","low"})
    return int(score>=6 or (x.get("HighBP")=="yes" and x.get("HighChol")=="yes" and bmi>=4) or (bmi>=4 and (x.get("HighBP")=="yes" or x.get("HighChol")=="yes") and (age>=2 or gh>=2 or x.get("DiffWalk")=="yes" or x.get("PhysActivity")=="no")) or (x.get("HeartDiseaseorAttack")=="yes" and (x.get("HighBP")=="yes" or x.get("HighChol")=="yes" or gh>=3 or x.get("DiffWalk")=="yes")) or (age>=4 and (x.get("HighBP")=="yes" or x.get("HighChol")=="yes" or x.get("HeartDiseaseorAttack")=="yes" or x.get("DiffWalk")=="yes" or x.get("PhysActivity")=="no" or gh>=2)))
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
