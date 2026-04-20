"""Auto-generated post-hoc boosted ensemble wrapper."""
from typing import Any

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

# Learner 1: row 0, attempt 1
def f(x):
    pm = float(x.get('profile_mean', 1e9))
    ps = float(x.get('profile_stdev', 1e9))
    psk = float(x.get('profile_skewness', 0.0))
    pk = float(x.get('profile_kurtosis', 0.0))
    dm = float(x.get('dm_snr_mean', 0.0))
    dms = float(x.get('dm_snr_stdev', 0.0))
    dmsk = float(x.get('dm_snr_skewness', 0.0))
    dmk = float(x.get('dm_snr_kurtosis', 0.0))

    # Strong non-pulsar / RFI patterns: very spiky DM curves with low DM mean.
    if (dm < 6.0 and (dmk > 80.0 or dmsk > 10.0)) or (dm < 2.5 and dmsk > 9.0):
        return 0
    if (psk < 0.3 and pk < 0.5 and dm < 6.0 and dmsk > 6.0):
        return 0

    # Pulsar-like: asymmetric, peaky profile + reasonably well-behaved DM curve.
    rule_a = (psk > 1.4 and pk > 4.0 and pm < 95.0 and dmsk < 4.8)

    # Pulsar-like: strong DM detection even if profile moments are weaker.
    rule_b = (dm > 12.0 and dmsk < 4.2 and dmk < 15.0 and pm < 130.0)

    # Softer catch-all for borderline positives.
    rule_c = (psk > 0.55 and pk > 0.0 and pm < 125.0 and ps < 60.0 and dm > 3.0 and dmsk < 7.2 and dmk < 60.0)

    return 1 if (rule_a or rule_b or rule_c) else 0
h_1 = f

# Learner 2: row 7, attempt 8
def f(x):
    pm = float(x.get('profile_mean', 1e9))
    ps = float(x.get('profile_stdev', 1e9))
    psk = float(x.get('profile_skewness', 0.0))
    pk = float(x.get('profile_kurtosis', 0.0))
    dm = float(x.get('dm_snr_mean', 0.0))
    dms = float(x.get('dm_snr_stdev', 0.0))
    dmsk = float(x.get('dm_snr_skewness', 1e9))
    dmk = float(x.get('dm_snr_kurtosis', 1e9))

    # Strong pulse-profile signature
    if pm <= 95 and psk >= 1.3 and pk >= 3.5:
        return 1

    # Weaker profile signature, but still pulsar-like
    if pm <= 110 and psk >= 1.1 and pk >= 2.5 and dmsk <= 6.5:
        return 1

    # Strong DM-SNR peak with at least some profile asymmetry
    if dm >= 25 and dmk <= 5 and dmsk <= 2.2 and psk >= 0.6:
        return 1

    # Very strong DM-SNR peak (many bright pulsars)
    if dm >= 80 and dmk <= 2 and psk >= 0.8:
        return 1

    # Special case: moderate DM peak + non-gaussian-ish profile kurtosis (filters many RFI)
    if pm > 110 and 12 <= dm < 20 and dmk < 12 and dmsk < 3.8 and pk > -0.35 and ps < 60:
        return 1

    return 0
h_2 = f

# Learner 3: row 8, attempt 9
def f(x):
    pm = float(x.get('profile_mean', 0.0))
    ps = float(x.get('profile_skewness', 0.0))
    pk = float(x.get('profile_kurtosis', 0.0))
    dm_m = float(x.get('dm_snr_mean', 0.0))
    dm_s = float(x.get('dm_snr_skewness', 0.0))
    dm_k = float(x.get('dm_snr_kurtosis', 0.0))

    # Very DM-peaky/erratic candidates are almost always non-pulsars
    if dm_k > 80.0 or dm_s > 10.0:
        return 0

    # Strong pulse-profile shape typical of pulsars
    if ps > 2.2 and pk > 7.0 and dm_m > 7.0 and dm_k < 60.0:
        return 1

    # Strong DM signature typical of pulsars (high mean, low skew/kurt)
    if dm_m > 75.0 and dm_s < 2.2 and dm_k < 2.5:
        return 1

    # Borderline pulsars: moderately pulsar-like profile + reasonable DM curve
    if ps > 1.4 and pk > 3.0 and dm_m > 15.0 and dm_s < 6.0 and dm_k < 40.0 and pm < 110.0:
        return 1

    return 0
h_3 = f

ALPHAS = [1, 1, 1]
DIRECTIONS = [1, 1, -1]
LEARNERS = [h_1, h_2, h_3]

def f(x: Any) -> int:
    score = 0.0
    for alpha, direction, learner in zip(ALPHAS, DIRECTIONS, LEARNERS):
        score += alpha * direction * _normalize_pred_to_pm1(learner(x))
    return 1 if score >= 0.0 else 0
