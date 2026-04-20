"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    """Heuristic rule for binned HTRU2 pulsar candidates.
    x: dict with keys like 'profile_mean', 'dm_snr_kurtosis', values in
       {'very low','low','medium','high','very high'}.
    Returns 0/1.
    """
    pm = x.get('profile_mean')
    psd = x.get('profile_stdev')
    psk = x.get('profile_skewness')
    pk = x.get('profile_kurtosis')

    dm_m = x.get('dm_snr_mean')
    dm_sd = x.get('dm_snr_stdev')
    dm_sk = x.get('dm_snr_skewness')
    dm_k = x.get('dm_snr_kurtosis')

    hi = {'high', 'very high'}
    lo = {'very low', 'low'}
    midlo = {'very low', 'low', 'medium'}

    # Pulsars typically have a strongly non-Gaussian pulse profile (high skew/kurt)
    # AND a clean DM-SNR signature (high mean/stdev, low skew/kurt).
    good_profile = (psk in hi) and (pk in hi) and (pm in midlo) and (psd != 'very high')
    good_dm = (dm_m in hi) and (dm_sd in hi) and (dm_sk in lo) and (dm_k in lo)

    return int(good_profile and good_dm)
h_1 = f

ALPHAS = [1.048570559389618]
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
