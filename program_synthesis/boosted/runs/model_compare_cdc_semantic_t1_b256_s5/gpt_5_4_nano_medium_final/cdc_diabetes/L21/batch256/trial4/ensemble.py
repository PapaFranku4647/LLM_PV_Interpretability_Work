"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def yes(v):
        return 1 if v == 'yes' else 0
    def lev(v, order):
        return order.get(v, 0)

    age = {'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    bmi  = {'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    gen  = {'very low':0,'low':1,'medium':2,'high':3,'very high':4,'very high ':4}
    hlth = {'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    edu  = {'very low':0,'low':1,'medium':2,'high':3,'very high':4}
    inc  = {'very low':0,'low':1,'medium':2,'high':3,'very high':4}

    risk = 0.0
    risk += 2.0 * yes(x.get('HighBP'))
    risk += 2.0 * yes(x.get('HighChol'))
    risk += 3.0 * yes(x.get('Stroke'))
    risk += 2.0 * yes(x.get('HeartDiseaseorAttack'))
    risk += 1.5 * yes(x.get('Smoker'))
    risk += 1.5 * yes(x.get('DiffWalk'))

    risk += 0.8 * lev(x.get('Age'), age)
    risk += 1.0 * lev(x.get('GenHlth'), gen)
    risk += 0.5 * lev(x.get('BMI'), bmi)
    risk += 0.25 * lev(x.get('MentHlth'), hlth)
    risk += 0.25 * lev(x.get('PhysHlth'), hlth)

    # behavior/access
    risk += 1.0 * (1 - yes(x.get('PhysActivity')))
    risk += 0.7 * yes(x.get('HvyAlcoholConsump'))
    risk -= 0.3 * yes(x.get('Fruits'))
    risk -= 0.3 * yes(x.get('Veggies'))

    # socioeconomic (lower -> higher risk)
    risk += 0.6 * (4 - lev(x.get('Education'), edu))
    risk += 0.4 * (4 - lev(x.get('Income'), inc))

    return 1 if risk >= 10.0 else 0
h_1 = f

ALPHAS = [0.4595133558465149]
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
