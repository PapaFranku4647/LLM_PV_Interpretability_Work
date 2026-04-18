"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    b = lambda k: str(x.get(k, 'false')).lower() == 'true'
    v = lambda k, d='none': str(x.get(k, d)).lower()

    # Strong "win" indicators in this feature set
    if b('stlmt'):
        return 1

    # Strong "no-win" cluster: black king in front of pawn + rook cannot help
    if b('bkxwp') and b('bkxbq') and b('bkxcr') and b('rimmx') and b('rkxwp'):
        return 0
    if b('bkxwp') and b('bkon8') and b('bkona') and b('bkspr'):
        return 0

    # Tactical win: active check with follow-up threat / resources
    if v('hdchk') == 'white' and (b('thrsk') or b('reskr') or b('reskd')):
        return 1

    # Positional win: contact + kings separated + white king/rook advantage flags
    if b('cntxt') and v('dsopp') == 'greater' and (b('blxwp') or b('wkovl') or b('wkcti') or b('reskr')):
        return 1

    # Endgame technique: rook can rescue + white king well placed, while black king not on pawn
    if b('reskr') and (b('wkna8') or b('wkovl') or b('wkcti')) and not b('bkxwp'):
        return 1

    # Default: prefer "no win" unless clear winning cues
    if b('bkxwp') and not (b('thrsk') or b('reskr') or v('hdchk') == 'white'):
        return 0

    return 1 if (b('thrsk') and (b('reskr') or b('reskd') or b('wkovl'))) else 0
h_1 = f

ALPHAS = [0.4033509400534038]
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
