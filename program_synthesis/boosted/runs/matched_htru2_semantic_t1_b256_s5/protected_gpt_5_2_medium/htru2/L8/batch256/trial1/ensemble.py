"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    r = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    pm = r[x["profile_mean"]]
    ps = r[x["profile_stdev"]]
    psk = r[x["profile_skewness"]]
    pk = r[x["profile_kurtosis"]]
    dm = r[x["dm_snr_mean"]]
    ds = r[x["dm_snr_stdev"]]
    dsk = r[x["dm_snr_skewness"]]
    dk = r[x["dm_snr_kurtosis"]]

    # Strong DM signature typical of pulsars: high mean/stdev but low skew/kurt.
    dm_pulsar_like = (dm >= 3 and ds >= 3 and dsk <= 1 and dk <= 1)

    # Profile shape typical of pulsars: high skew/kurt with low baseline.
    profile_pulsar_like = (psk >= 3 and pk >= 2 and pm <= 1)

    # Very strong, clean pulsar profile (helps when DM is only moderate).
    profile_very_strong = (psk == 4 and pk >= 3 and pm <= 1 and ps <= 1 and dm >= 2 and ds >= 2)

    return int((dm_pulsar_like and profile_pulsar_like) or profile_very_strong)
h_1 = f

ALPHAS = [1.207531538210368]
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
