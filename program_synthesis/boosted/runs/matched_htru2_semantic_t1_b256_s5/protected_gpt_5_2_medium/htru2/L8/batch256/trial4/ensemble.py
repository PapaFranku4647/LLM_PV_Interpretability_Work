"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule for binned HTRU2 features."""
    dm_strong = (
        x.get("dm_snr_mean") in ("high", "very high")
        and x.get("dm_snr_stdev") in ("high", "very high")
        and x.get("dm_snr_skewness") in ("very low", "low")
        and x.get("dm_snr_kurtosis") in ("very low", "low")
    )

    prof_pulsarish = (
        x.get("profile_skewness") in ("high", "very high")
        and x.get("profile_kurtosis") in ("medium", "high", "very high")
    )

    return 1 if (dm_strong and prof_pulsarish) else 0
h_1 = f

ALPHAS = [1.261810580984345]
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
