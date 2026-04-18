"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    def s(k):
        return (x.get(k) or "")
    def z(k, d=0.0):
        try:
            return float(x.get(k, d))
        except Exception:
            return float(d)

    cap_color = s("cap_color")
    cap_surface = s("cap_surface")
    gatt = s("gill_attachment")
    gcol = s("gill_color")
    bruises = s("does_bruise_or_bleed")
    ring = s("has_ring")
    rtype = s("ring_type")
    hab = s("habitat")
    sp = s("spore_print_color")

    # strong poisonous cues
    if rtype.startswith("none"):
        return 0
    if gcol.startswith("green"):
        return 0
    if cap_color.startswith("green") and bruises.startswith("yes"):
        return 0
    if gatt.startswith("pores") and gcol.startswith("red"):
        return 0
    if cap_color.startswith("red") and gatt.startswith("pores") and bruises.startswith("no"):
        return 0

    # gill-less / puffball-like: gray spore print tends edible
    if gatt.startswith("none"):
        return 1 if sp.startswith("gray") else 0

    sw = z("stem_width_z")

    # robust edible cues
    if sw > 1.0:
        return 1
    if ring.startswith("yes") and ("evanescent" in rtype or "zone" in rtype or "pendant" in rtype or "movable" in rtype or "grooved" in rtype or "large" in rtype):
        return 1
    if gatt.startswith("pores"):
        return 1

    # common edible white-gilled, non-bruising grass/meadow types
    if cap_color.startswith("white") and gcol.startswith("white") and bruises.startswith("no") and ("meadows" in hab or "grasses" in hab) and z("stem_height_z") < 0.2:
        return 1

    return 1 if sw > 0.3 else 0
h_1 = f

ALPHAS = [0.4595133558465149]
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
