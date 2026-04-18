"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    hi = {"high", "very high"}
    lo = {"very low", "low"}

    dm_strong = (
        x.get("dm_snr_mean") in hi
        and x.get("dm_snr_stdev") in hi
        and x.get("dm_snr_skewness") in lo
        and x.get("dm_snr_kurtosis") in lo
    )

    prof_pulsarish = (
        x.get("profile_skewness") in hi
        and x.get("profile_kurtosis") in hi
    )

    # Main rule: strong DM response + pulsar-like pulse profile shape
    if dm_strong and prof_pulsarish:
        return 1

    # Secondary rule: very pulse-like profile can still indicate pulsar when DM is not clearly RFI-like
    if prof_pulsarish and x.get("dm_snr_skewness") not in {"high", "very high"} and x.get("dm_snr_kurtosis") not in {"high", "very high"}:
        return 1

    return 0
h_1 = f

ALPHAS = [1.23404976573581]
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
