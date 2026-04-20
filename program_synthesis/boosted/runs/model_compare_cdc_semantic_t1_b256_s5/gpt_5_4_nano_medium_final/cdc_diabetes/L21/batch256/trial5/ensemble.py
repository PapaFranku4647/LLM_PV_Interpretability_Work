"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def yn(k, default='no'):
        return (x.get(k, default) or '').strip().lower() == 'yes'

    order = {
        'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4
    }

    def ordv(k):
        v = (x.get(k, '') or '').strip().lower()
        return order.get(v, 2)  # neutral if missing/unknown

    # Risk score (hand-crafted heuristic)
    s = 0.0

    # Strong clinical risk factors
    if yn('HighBP'): s += 2.0
    if yn('HighChol'): s += 2.0
    if yn('Stroke'): s += 3.0
    if yn('HeartDiseaseorAttack'): s += 3.0
    if yn('DiffWalk'): s += 2.5
    if yn('Smoker'): s += 1.5

    # Access / adherence proxies
    if yn('NoDocbcCost'): s += 1.5
    if (x.get('AnyHealthcare', '') or '').strip().lower() == 'no':
        s += 1.0

    # Lifestyle risk/protective
    if yn('HvyAlcoholConsump'): s += 1.0
    if yn('PhysActivity'): s -= 1.2
    if yn('Fruits'): s -= 0.8
    if yn('Veggies'): s -= 0.8

    # Ordinal health/burden
    s += 0.9 * ordv('BMI')
    s += 0.7 * ordv('Age')
    s += 1.0 * ordv('GenHlth')
    s += 0.3 * ordv('MentHlth')
    s += 0.3 * ordv('PhysHlth')

    # Socio-demographic (protective when education/income are high)
    # higher ordinal => more protective, so risk decreases with ordv
    s += 0.3 * (4 - ordv('Education'))
    s += 0.25 * (4 - ordv('Income'))

    # Mild sex adjustment
    if (x.get('Sex', '') or '').strip().lower() == 'male':
        s += 0.2

    return 1 if s >= 7.0 else 0
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
