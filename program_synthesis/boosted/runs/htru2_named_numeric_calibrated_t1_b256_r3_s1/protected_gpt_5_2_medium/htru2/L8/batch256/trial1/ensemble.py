"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    pm = float(x.get('profile_mean', 999.0))
    pst = float(x.get('profile_stdev', 999.0))
    ps = float(x.get('profile_skewness', 0.0))
    pk = float(x.get('profile_kurtosis', 0.0))
    dm = float(x.get('dm_snr_mean', 0.0))
    ds = float(x.get('dm_snr_stdev', 0.0))
    dms = float(x.get('dm_snr_skewness', 999.0))
    dmk = float(x.get('dm_snr_kurtosis', 999.0))

    # obvious non-pulsar region (very spiky DM tails)
    if dms > 12 or dmk > 150:
        return 0

    score = (pm < 100) + (ps > 1.1) + (pk > 2.5) + (dm > 10) + (ds > 30) + (dms < 6) + (dmk < 60)

    strong_profile = (pm < 85 and ps > 1.8 and pk > 6)
    strong_dm = (dm > 25 and ds > 50 and dms < 3 and dmk < 15 and pst < 50)

    return int(score >= 5 or strong_profile or strong_dm)
h_1 = f

ALPHAS = [1.157772118818275]
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
