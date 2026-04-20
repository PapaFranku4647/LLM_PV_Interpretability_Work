"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule for binned HTRU2 pulsar candidate features.
    Input: dict mapping feature name -> one of {'very low','low','medium','high','very high'}.
    Output: 1 (pulsar) or 0 (non-pulsar).
    """
    rank = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    def v(k, default="medium"):
        return rank.get(x.get(k, default), 2)

    pm = v("profile_mean")
    ps = v("profile_stdev")
    psk = v("profile_skewness")
    pku = v("profile_kurtosis")

    dm = v("dm_snr_mean")
    ds = v("dm_snr_stdev")
    dsk = v("dm_snr_skewness")
    dku = v("dm_snr_kurtosis")

    # Strong non-pulsar signature: low DM mean + very spiky/asymmetric DM curve
    if dm <= 1 and dsk >= 3 and dku >= 3:
        return 0

    # Main pulsar signature: sharp/narrow pulse profile + strong DM mean/stdev + low DM skew/kurt
    score = 0
    score += (pm <= 1)
    score += (ps <= 1)
    score += (psk >= 3)
    score += (pku >= 3)
    score += (dm >= 3)
    score += (ds >= 3)
    score += (dsk <= 1)
    score += (dku <= 1)
    if score >= 6:
        return 1

    # Secondary: extremely peaked profile can be pulsar even if DM is only moderate
    if psk >= 3 and pku >= 4 and pm <= 2 and dm >= 2 and dsk <= 2:
        return 1

    # Secondary: very strong DM (very high mean + high stdev) with not-too-bad profile
    if dm == 4 and ds >= 3 and dsk <= 2 and psk >= 2 and pm <= 3:
        return 1

    return 0
h_1 = f

ALPHAS = [1.068942568774955]
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
