"""Auto-generated boosted ensemble wrapper."""
from typing import Any

# Learner 1
def f(x):
    attach = x.get("gill_attachment")
    gcol = x.get("gill_color")
    spore = x.get("spore_print_color")
    ring = x.get("has_ring")
    rtype = x.get("ring_type")
    bruis = x.get("does_bruise_or_bleed")
    cap_col = x.get("cap_color")
    habitat = x.get("habitat")
    stem_root = x.get("stem_root")

    # strong poisonous cues in this dataset slice
    if spore == "pink" or gcol == "green" or rtype == "none":
        return 0

    # pore-bearing mushrooms are mostly edible here, except a few clear patterns
    if attach == "pores":
        if ring == "yes" or stem_root == "club" or gcol in {"red", "pink"}:
            return 0
        return 1

    # "no gills" cases (puffball-like): often edible when no ring and spore not pink
    if attach == "none":
        if ring == "no" and spore in {"gray", "white", "unknown"} and cap_col in {"brown", "yellow", "white"}:
            return 1
        return 0

    # ring types that frequently indicate edible in the examples
    if ring == "yes" and rtype in {"pendant", "zone", "large", "movable", "grooved", "evanescent"}:
        if cap_col != "green" and gcol not in {"green"}:
            return 1

    # many edible examples share white gills, no bruising, and typical habitats
    if gcol == "white" and bruis == "no" and habitat in {"woods", "meadows", "leaves", "paths"}:
        return 1

    return 0
h_1 = f

ALPHAS = [0.5084671288269212]
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
