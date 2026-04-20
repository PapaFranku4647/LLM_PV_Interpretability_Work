"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    o = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}

    dm_mean = x.get("dm_snr_mean")
    dm_sd   = x.get("dm_snr_stdev")
    dm_sk   = x.get("dm_snr_skewness")
    dm_ku   = x.get("dm_snr_kurtosis")
    pr_mean = x.get("profile_mean")
    pr_sk   = x.get("profile_skewness")
    pr_ku   = x.get("profile_kurtosis")

    # Strong RFI-like signature in DM statistics
    if dm_sk in ("high", "very high") and dm_ku in ("high", "very high"):
        return 0

    dm_good = (dm_sk in ("very low", "low")) and (dm_ku in ("very low", "low"))
    pr_ok = (pr_sk not in ("very low", "low")) and (pr_ku not in ("very low", "low"))

    # Typical pulsar-like pattern: very strong DM SNR with low DM skew/kurt and a non-flat profile
    if dm_good and pr_ok and dm_mean == "very high" and dm_sd in ("high", "very high"):
        return 1

    # Fallback: simple monotone score (high DM mean/stdev + peaky profile, penalize DM skew/kurt)
    score = o.get(dm_mean, 0) + o.get(dm_sd, 0) + o.get(pr_sk, 0) + o.get(pr_ku, 0) - o.get(dm_sk, 0) - o.get(dm_ku, 0)

    return 1 if (dm_good and score >= 13 and pr_mean != "very high") else 0
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
