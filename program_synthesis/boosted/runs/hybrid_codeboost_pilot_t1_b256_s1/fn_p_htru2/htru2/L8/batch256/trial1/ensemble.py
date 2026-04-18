"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def g(k, default=0.0):
        try:
            v = x.get(k, default)
            return float(v)
        except Exception:
            return float(default)

    pmean = g('profile_mean_z')
    pskew = g('profile_skewness_z')
    pkurt = g('profile_kurtosis_z')

    dmmean = g('dm_snr_mean_z')
    dmstd = g('dm_snr_stdev_z')
    dmskew = g('dm_snr_skewness_z')
    dmkurt = g('dm_snr_kurtosis_z')

    pskew_bin = (x.get('profile_skewness_bin') or '').strip().lower()
    pkurt_bin = (x.get('profile_kurtosis_bin') or '').strip().lower()

    # Strong pulsar-like pulse profile shape (dominant signal in these examples)
    if pskew > 1.6 and pkurt > 0.9:
        return 1

    # Very low mean with noticeably heavy-tailed / skewed profile
    if pmean < -2.0 and pskew > 0.4 and pkurt > 0.0:
        return 1
    if pmean < -1.0 and pskew > 0.7 and pkurt > 0.2:
        return 1

    # Use bins as readable guards (when bins indicate extreme shape)
    if pmean < -0.8 and pskew_bin == 'very high' and pkurt_bin in ('high', 'very high'):
        return 1

    # DM SNR pattern can support a borderline pulse-profile decision
    if pmean < -0.5 and pskew > 0.6 and dmstd > 1.6 and dmskew < -1.2 and dmkurt < -0.9 and dmmean > 0.0:
        return 1

    return 0
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
