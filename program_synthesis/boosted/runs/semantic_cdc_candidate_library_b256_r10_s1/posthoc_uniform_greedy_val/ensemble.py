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

# Learner 1: row 8, attempt 9
def f(x):
    # Rule-based risk score for diabetes (0/1) using key CDC indicators.
    yn = lambda k: (x.get(k, "no") == "yes")

    ord5 = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    bmi = x.get("BMI", "medium")
    age = x.get("Age", "medium")
    gen = x.get("GenHlth", "medium")
    inc = x.get("Income", "medium")
    edu = x.get("Education", "medium")

    score = 0.0

    # Strong clinical correlates
    score += 2.0 * yn("HighBP")
    score += 1.5 * yn("HighChol")
    score += 2.0 * yn("Stroke")
    score += 2.0 * yn("HeartDiseaseorAttack")
    score += 1.5 * yn("DiffWalk")

    # Anthropometrics and age
    score += {"very low": -0.5, "low": 0.0, "medium": 0.5, "high": 1.5, "very high": 2.5}.get(bmi, 0.5)
    score += {"very low": 0.0, "low": 0.5, "medium": 1.0, "high": 1.7, "very high": 2.3}.get(age, 1.0)

    # Self-rated health (worse -> higher risk)
    score += 0.6 * ord5.get(gen, 2)

    # Lifestyle and access (smaller effects)
    score += 0.5 * yn("Smoker")
    score -= 1.0 * yn("PhysActivity")
    score -= 0.3 * yn("HvyAlcoholConsump")

    # Socioeconomic gradient (lower -> higher risk)
    score += 0.25 * (2 - min(2, ord5.get(inc, 2)))
    score += 0.15 * (2 - min(2, ord5.get(edu, 2)))

    # Require at least one major metabolic/cardiovascular flag unless score is very high
    major_flag = yn("HighBP") or yn("HighChol") or (bmi in ("high", "very high"))

    return int((score >= 6.0 and major_flag) or (score >= 7.5))
h_1 = f

# Learner 2: row 6, attempt 7
def f(x):
    # Simple risk-score rule for diabetes indicators (deterministic, non-trainable)
    ord_map = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
    def o(k):
        return ord_map.get(str(x.get(k, "")).strip().lower(), 2)
    def yn(k):
        return str(x.get(k, "no")).strip().lower() == "yes"

    bmi = str(x.get("BMI", "medium")).strip().lower()
    age = o("Age")
    gen = o("GenHlth")

    risk = 0

    # Major cardiometabolic risks
    risk += 2 if yn("HighBP") else 0
    risk += 2 if yn("HighChol") else 0
    risk += 3 if bmi in {"high", "very high"} else (1 if bmi == "medium" else 0)
    risk += 2 if age >= 3 else (1 if age == 2 else 0)

    # General health / functional limitation (often proxies for chronic illness)
    risk += 2 if gen >= 3 else (1 if gen == 2 else 0)
    risk += 1 if yn("DiffWalk") else 0

    # Vascular events / CVD
    risk += 2 if yn("HeartDiseaseorAttack") else 0
    risk += 2 if yn("Stroke") else 0

    # Lifestyle / socioeconomic modifiers
    risk += 1 if yn("Smoker") else 0
    risk -= 1 if yn("PhysActivity") else 0
    if yn("Fruits") and yn("Veggies"):
        risk -= 1
    inc = o("Income")
    risk += 1 if inc <= 1 else 0

    return 1 if risk >= 7 else 0
h_2 = f

# Learner 3: row 0, attempt 1
def f(x):
    """Heuristic, non-trainable rule for CDC diabetes indicator (1=yes diabetes, 0=no)."""
    g = lambda k, d="": (x.get(k, d) or d)

    highbp = (g("HighBP") == "yes")
    highchol = (g("HighChol") == "yes")
    stroke = (g("Stroke") == "yes")
    heart = (g("HeartDiseaseorAttack") == "yes")
    diffwalk = (g("DiffWalk") == "yes")

    bmi = g("BMI")
    age = g("Age")
    gen = g("GenHlth")

    obese = bmi in {"high", "very high"}
    older = age in {"high", "very high"}
    very_old = age == "very high"
    poor_gen = gen in {"very low", "low"}
    comorb = stroke or heart
    metabolic = highbp or highchol

    # Gate: without obesity/older-age/comorbidity, usually predict 0
    if not (obese or older or comorb):
        return 0

    # Obesity-driven diabetes
    if obese:
        if (highbp and highchol):
            return 1
        if metabolic and (older or very_old or poor_gen or diffwalk or comorb):
            return 1
        return 0

    # Non-obese: require substantial vascular/mobility signal
    if comorb and metabolic:
        # allow even if not old; comorbidity + HTN/Chol is a strong signal
        return 1

    if diffwalk and highbp and older and (highchol or poor_gen):
        return 1

    if highbp and highchol and very_old and poor_gen:
        return 1

    return 0
h_3 = f

# Learner 4: row 31, attempt 32
def f(x):
    # Heuristic rule (matches labels in provided samples): output 0 for high cardio-metabolic burden, else 1.
    v = lambda k, d=None: x.get(k, d)

    score = 0

    if v("HighBP") == "yes": score += 2
    if v("HighChol") == "yes": score += 1

    bmi = v("BMI")
    if bmi == "high": score += 1
    elif bmi == "very high": score += 2

    age = v("Age")
    if age == "high": score += 1
    elif age == "very high": score += 2

    if v("Smoker") == "yes": score += 1
    if v("Stroke") == "yes": score += 2
    if v("HeartDiseaseorAttack") == "yes": score += 2
    if v("DiffWalk") == "yes": score += 2

    gen = v("GenHlth")
    if gen == "high": score += 1
    elif gen == "very high": score += 2

    ph = v("PhysHlth")
    if ph == "high": score += 1
    elif ph == "very high": score += 2

    # Threshold chosen to make "severe" profiles map to 0, otherwise 1.
    return 0 if score >= 6 else 1
h_4 = f

# Learner 5: row 29, attempt 30
def f(x):
    # simple rule-based diabetes risk score (no training)
    yn = lambda k: 1 if x.get(k) == 'yes' else 0

    # ordinal bins: very low < low < medium < high < very high
    ord_map = {'very low': 0.0, 'low': 0.5, 'medium': 1.0, 'high': 1.5, 'very high': 2.0}
    def o(k):
        return ord_map.get(x.get(k), 0.0)

    bmi = x.get('BMI')
    if bmi in ('very high', 'high'):
        bmi_s = 2.0
    elif bmi == 'medium':
        bmi_s = 1.0
    else:
        bmi_s = 0.0

    age = x.get('Age')
    if age in ('very high', 'high'):
        age_s = 2.0
    elif age == 'medium':
        age_s = 1.0
    else:
        age_s = 0.0

    score = 0.0
    score += 2.0 * yn('HighBP')
    score += 1.5 * yn('HighChol')
    score += 2.0 * yn('Stroke')
    score += 2.0 * yn('HeartDiseaseorAttack')
    score += 1.5 * yn('DiffWalk')
    score += 1.0 * (1 - yn('PhysActivity'))  # inactivity
    score += 0.5 * yn('Smoker')
    score += 1.0 * o('GenHlth')
    score += bmi_s
    score += age_s

    # mild socioeconomic signal
    inc = x.get('Income')
    if inc in ('very low', 'low'):
        score += 0.5

    return 1 if score >= 5.0 else 0
h_5 = f

# Learner 6: row 36, attempt 37
def f(x):
    """Heuristic, non-trainable approximation for CDC diabetes indicator (binary).
    Expects x as a dict of feature->string (e.g., 'yes'/'no', 'low'..'very high').
    Returns 0/1.
    """

    def lvl(v):
        m = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
        return m.get((v or "").strip().lower(), 2)

    def is_yes(k):
        return (x.get(k, "").strip().lower() == "yes")

    score = 0

    # Core metabolic/cardiovascular risk
    if is_yes("HighBP"):
        score += 2
    if is_yes("HighChol"):
        score += 2

    b = lvl(x.get("BMI"))
    if b == 4:
        score += 2
    elif b == 3:
        score += 1

    a = lvl(x.get("Age"))
    if a == 4:
        score += 2
    elif a >= 2:
        score += 1

    # Comorbidities / functional limitation
    if is_yes("Stroke"):
        score += 2
    if is_yes("HeartDiseaseorAttack"):
        score += 2
    if is_yes("DiffWalk"):
        score += 2

    # Lifestyle
    if is_yes("Smoker"):
        score += 1
    if (x.get("PhysActivity", "").strip().lower() == "no"):
        score += 1

    # Self-reported health: treat worse health as "very low/low" (common discretization of GenHlth)
    g = (x.get("GenHlth") or "").strip().lower()
    if g == "very low":
        score += 2
    elif g == "low":
        score += 1

    # PhysHlth: higher is worse
    p = lvl(x.get("PhysHlth"))
    if p == 4:
        score += 2
    elif p == 3:
        score += 1

    # Socioeconomic small effect
    inc = lvl(x.get("Income"))
    edu = lvl(x.get("Education"))
    if inc <= 1:
        score += 1
    if edu <= 1:
        score += 1

    # Empirical safeguard seen in this sample: heavy comorbidity clusters often labeled 0
    if is_yes("Stroke") and is_yes("HeartDiseaseorAttack"):
        score -= 3

    return 1 if score >= 5 else 0
h_6 = f

ALPHAS = [1, 1, 1, 1, 1, 1]
DIRECTIONS = [1, 1, -1, -1, 1, -1]
LEARNERS = [h_1, h_2, h_3, h_4, h_5, h_6]

def f(x: Any) -> int:
    score = 0.0
    for alpha, direction, learner in zip(ALPHAS, DIRECTIONS, LEARNERS):
        score += alpha * direction * _normalize_pred_to_pm1(learner(x))
    return 1 if score >= 0.0 else 0
