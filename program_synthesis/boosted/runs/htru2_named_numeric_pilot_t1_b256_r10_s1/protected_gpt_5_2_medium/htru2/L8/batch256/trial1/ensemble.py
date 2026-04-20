"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
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
