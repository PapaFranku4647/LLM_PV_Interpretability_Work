"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    pm = float(x.get('profile_mean', 999.0))
    ps = float(x.get('profile_skewness', 0.0))
    pk = float(x.get('profile_kurtosis', 0.0))
    dm = float(x.get('dm_snr_mean', 0.0))
    dms = float(x.get('dm_snr_skewness', 0.0))
    dmk = float(x.get('dm_snr_kurtosis', 0.0))

    # very strong non-pulsar DM signature (tiny mean, huge skew/kurtosis)
    if dm < 6 and dms > 8 and dmk > 40:
        return 0

    # strong pulsar-like pulse profile shape
    if ps > 1.8 and pk > 6 and pm < 120:
        return 1

    # strong pulsar-like DM curve (high mean, low skew/kurtosis) + not-flat profile
    if dm > 20 and dms < 3.5 and dmk < 15 and pm < 115 and ps > 0.5:
        return 1

    # very high DM mean sometimes indicates pulsar even with modest profile stats
    if dm > 60 and dmk < 5 and pm < 100:
        return 1

    # fallback: simple vote
    score = 0
    score += (ps > 1.2)
    score += (pk > 4)
    score += (dm > 10)
    score += (dms < 6)
    score += (dmk < 30)
    score += (pm < 110)
    return 1 if score >= 4 else 0
h_1 = f

ALPHAS = [1.290949457887655]
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
