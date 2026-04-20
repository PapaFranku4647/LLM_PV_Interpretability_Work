"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    pm=float(x.get('profile_mean',0))
    psk=float(x.get('profile_skewness',0))
    pk=float(x.get('profile_kurtosis',0))
    dm=float(x.get('dm_snr_mean',0))
    dsk=float(x.get('dm_snr_skewness',99))
    dk=float(x.get('dm_snr_kurtosis',99))
    return int((pm<100 and psk>1.25 and pk>2.8) or (dm>10.5 and dsk<3.5 and dk<15 and psk>0.55))
h_1 = f

ALPHAS = [1.464056042939506]
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
