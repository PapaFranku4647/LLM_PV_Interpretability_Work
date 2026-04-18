"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    o={"very low":0,"low":1,"medium":2,"high":3,"very high":4}
    g=x.get
    bmi=o.get(g("BMI"),0)
    age=o.get(g("Age"),0)
    gen=o.get(g("GenHlth"),0)
    phys=o.get(g("PhysHlth"),0)
    bp=g("HighBP")=="yes"
    chol=g("HighChol")=="yes"
    heart=g("HeartDiseaseorAttack")=="yes"
    stroke=g("Stroke")=="yes"
    diff=g("DiffWalk")=="yes"
    inactive=g("PhysActivity")=="no"
    smoker=g("Smoker")=="yes"
    low_ses=o.get(g("Income"),4)<=1 or o.get(g("Education"),4)<=1

    if diff and (bp or chol or heart or stroke or bmi>=3 or gen>=3 or phys>=3):
        return 1
    if stroke and (bp or chol or gen>=2 or phys>=2 or age>=3):
        return 1
    if bp and chol and age>=3 and bmi<=1 and not (diff or heart or stroke) and not inactive and (gen==2 or phys>=3):
        return 0
    if bp and age>=3 and phys<=1 and not (stroke or diff):
        return 1

    s=0
    s+=2*bp+chol+smoker+inactive+low_ses+heart
    s+=2 if bmi>=3 else 1 if bmi==2 else 0
    s+=2 if gen>=3 else 0
    s+=1 if phys>=3 else 0
    s+=1 if age>=3 else 0
    s+=1 if bp and age>=3 and (chol or bmi>=2 or low_ses) else 0
    s+=2 if age>=3 and chol and smoker else 0
    s+=1 if chol and bmi>=3 and (age>=3 or smoker) else 0
    return int(s>=5)
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
