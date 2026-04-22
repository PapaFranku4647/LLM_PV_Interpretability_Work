import json, glob, os

root = r'C:\Users\lucas\Desktop\Coding\TomerResearch\LLM_PV_Working_Copy\program_synthesis\runs_code0_prompt_comparison\20260303_115540'

variants = ["standard", "explain", "interview", "preview", "multipath",
            "subgroups", "thesis_aware", "regional", "ensemble"]

# ============================================================
# Load LOW reasoning (2 seeds done separately)
# ============================================================
low_data = {}
for v in variants:
    low_data[v] = {"cases": [], "summaries": []}
    for seed_dir in ["low", "low_seed2202"]:
        sfiles = glob.glob(os.path.join(root, seed_dir, v, '*', 'overall_summary.json'))
        cfiles = glob.glob(os.path.join(root, seed_dir, v, '*', 'cases.jsonl'))
        if sfiles:
            with open(sfiles[0]) as f:
                low_data[v]["summaries"].append(json.load(f))
        if cfiles:
            with open(cfiles[0]) as f:
                for line in f:
                    low_data[v]["cases"].append(json.loads(line))

# ============================================================
# Load MEDIUM reasoning (both seeds in one run)
# ============================================================
med_data = {}
for v in variants:
    med_data[v] = {"cases": [], "summaries": []}
    sfiles = glob.glob(os.path.join(root, "medium", v, '*', 'overall_summary.json'))
    cfiles = glob.glob(os.path.join(root, "medium", v, '*', 'cases.jsonl'))
    if sfiles:
        with open(sfiles[0]) as f:
            med_data[v]["summaries"].append(json.load(f))
    if cfiles:
        with open(cfiles[0]) as f:
            for line in f:
                med_data[v]["cases"].append(json.loads(line))

# ============================================================
# Compute metrics from cases directly (most accurate)
# ============================================================
def compute_metrics(cases):
    if not cases:
        return {}
    n = len(cases)

    # Code0 accuracies (per combo, not per sample)
    combos = {}
    for c in cases:
        key = (c["fn"], c["seed"])
        if key not in combos:
            combos[key] = c

    train_accs = [x["train_acc"] for x in combos.values() if x.get("train_acc") is not None]
    val_accs = [x["val_acc"] for x in combos.values() if x.get("val_acc") is not None]
    test_accs = [x["test_acc"] for x in combos.values() if x.get("test_acc") is not None]

    cov_vals = [c["coverage_eq"] for c in cases]
    fc0_defined = [c["faithfulness_code0"] for c in cases if c.get("faithfulness_code0") is not None]
    fc0_all = [c["faithfulness_code0"] if c.get("faithfulness_code0") is not None else 0 for c in cases]
    fgt_all = [c["faithfulness_gt"] if c.get("faithfulness_gt") is not None else 0 for c in cases]
    accepted = sum(1 for c in cases if c.get("code1_accepted")) / n
    compile_ok = sum(1 for c in cases if c.get("code1_compile_ok")) / n
    missing_cond = sum(1 for c in cases if c.get("code1_verification_error") == "missing_conditions")
    a_s_sizes = [c["A_S_size"] for c in cases if c.get("A_S_size") is not None]

    return {
        "n": n,
        "train_acc": sum(train_accs)/len(train_accs) if train_accs else None,
        "val_acc": sum(val_accs)/len(val_accs) if val_accs else None,
        "test_acc": sum(test_accs)/len(test_accs) if test_accs else None,
        "cov_eq": sum(cov_vals)/len(cov_vals) if cov_vals else None,
        "fc0": sum(fc0_all)/n,
        "fgt": sum(fgt_all)/n,
        "accepted": accepted,
        "missing_cond": missing_cond,
        "A_S_size": sum(a_s_sizes)/len(a_s_sizes) if a_s_sizes else None,
    }

low_metrics = {v: compute_metrics(low_data[v]["cases"]) for v in variants}
med_metrics = {v: compute_metrics(med_data[v]["cases"]) for v in variants}

# Also per-dataset
low_fn = {}
med_fn = {}
for fn_key in ["fn_n", "fn_o"]:
    low_fn[fn_key] = {v: compute_metrics([c for c in low_data[v]["cases"] if c["fn"] == fn_key]) for v in variants}
    med_fn[fn_key] = {v: compute_metrics([c for c in med_data[v]["cases"] if c["fn"] == fn_key]) for v in variants}

def fv(val, w=9):
    if val is None: return ("%"+str(w)+"s") % "n/a"
    return ("%"+str(w)+".3f") % val

def delta_str(low_val, med_val, w=7):
    if low_val is None or med_val is None:
        return ("%"+str(w)+"s") % "n/a"
    d = med_val - low_val
    sign = "+" if d >= 0 else ""
    return ("%"+str(w)+"s") % ("%s%.3f" % (sign, d))

# ============================================================
# MAIN TABLE: Low vs Medium side by side
# ============================================================
def composite(m):
    cov = m.get("cov_eq", 0) or 0
    fc0 = m.get("fc0", 0) or 0
    ta = m.get("train_acc", 0) or 0
    return 0.4*cov + 0.4*fc0 + 0.2*ta

# Sort by medium composite
scored_med = sorted(variants, key=lambda v: composite(med_metrics[v]), reverse=True)
scored_low = sorted(variants, key=lambda v: composite(low_metrics[v]), reverse=True)

print("=" * 130)
print("  LOW vs MEDIUM REASONING — Full Comparison (2 seeds x 2 datasets x 10 samples = 40 cases per variant)")
print("=" * 130)

metrics_list = [
    ("train_acc", "train_acc"),
    ("val_acc",   "val_acc"),
    ("test_acc",  "test_acc"),
    ("cov_eq",    "cov_eq"),
    ("faith_c0",  "fc0"),
    ("faith_gt",  "fgt"),
    ("accepted",  "accepted"),
    ("A_S_size",  "A_S_size"),
]

# Print by medium ranking
print("\n%-14s" % "" + "".join("  %-23s" % v for v in scored_med))
print("%-14s" % "metric" + "".join("  %9s %9s %7s" % ("low", "med", "delta") for _ in scored_med))
print("-" * (14 + 25 * len(scored_med)))

for label, key in metrics_list:
    parts = ["%-14s" % label]
    for v in scored_med:
        lv = low_metrics[v].get(key)
        mv = med_metrics[v].get(key)
        parts.append("  %s %s %s" % (fv(lv), fv(mv), delta_str(lv, mv)))
    print("".join(parts))

# Missing conditions row
parts = ["%-14s" % "miss_cond"]
for v in scored_med:
    lv = low_metrics[v].get("missing_cond", 0)
    mv = med_metrics[v].get("missing_cond", 0)
    parts.append("  %9d %9d %7s" % (lv, mv, "%+d" % (mv - lv)))
print("".join(parts))

# ============================================================
# PER-DATASET Low vs Medium
# ============================================================
for fn_key, fn_label in [("fn_n", "MUSHROOM"), ("fn_o", "DIABETES")]:
    print("\n--- %s: Low vs Medium ---" % fn_label)
    print("%-14s" % "metric" + "".join("  %-23s" % v for v in scored_med))
    print("%-14s" % "" + "".join("  %9s %9s %7s" % ("low", "med", "delta") for _ in scored_med))
    print("-" * (14 + 25 * len(scored_med)))
    for label, key in metrics_list:
        parts = ["%-14s" % label]
        for v in scored_med:
            lv = low_fn[fn_key][v].get(key)
            mv = med_fn[fn_key][v].get(key)
            parts.append("  %s %s %s" % (fv(lv), fv(mv), delta_str(lv, mv)))
        print("".join(parts))

# ============================================================
# RANKING COMPARISON
# ============================================================
print("\n" + "=" * 130)
print("  RANKING COMPARISON: Low vs Medium")
print("=" * 130)

print("\n  Composite ranking (0.4*cov + 0.4*fc0 + 0.2*train):")
print("    %3s  %-14s %8s %5s  |  %-14s %8s %5s  |  %s" % (
    "#", "LOW variant", "comp", "rank", "MED variant", "comp", "rank", "movement"))
print("    " + "-" * 95)

low_rank = {v: i+1 for i, v in enumerate(scored_low)}
med_rank = {v: i+1 for i, v in enumerate(scored_med)}

for rank in range(1, 10):
    lv = scored_low[rank-1]
    mv = scored_med[rank-1]
    lcomp = composite(low_metrics[lv])
    mcomp = composite(med_metrics[mv])

    # Movement for the medium variant
    movement = low_rank.get(mv, "?") - rank
    if movement > 0:
        mov_str = "up %d" % movement
    elif movement < 0:
        mov_str = "down %d" % abs(movement)
    else:
        mov_str = "same"

    print("    %3d  %-14s %8.4f %5d  |  %-14s %8.4f %5d  |  %s" % (
        rank, lv, lcomp, rank, mv, mcomp, med_rank[lv], mov_str))

# ============================================================
# DELTAS SUMMARY
# ============================================================
print("\n" + "=" * 130)
print("  AVERAGE EFFECT OF MEDIUM vs LOW REASONING (across all 9 variants)")
print("=" * 130)

for label, key in metrics_list:
    deltas = []
    for v in variants:
        lv = low_metrics[v].get(key)
        mv = med_metrics[v].get(key)
        if lv is not None and mv is not None:
            deltas.append(mv - lv)
    if deltas:
        avg_d = sum(deltas) / len(deltas)
        min_d = min(deltas)
        max_d = max(deltas)
        sign = "+" if avg_d >= 0 else ""
        print("  %-14s avg_delta=%s%.3f  (range: %+.3f to %+.3f)" % (label, sign, avg_d, min_d, max_d))

# Total missing conditions
total_low_miss = sum(low_metrics[v].get("missing_cond", 0) for v in variants)
total_med_miss = sum(med_metrics[v].get("missing_cond", 0) for v in variants)
print("\n  missing_conditions: low=%d  medium=%d" % (total_low_miss, total_med_miss))

# ============================================================
# FINAL COMPOSITE TABLE
# ============================================================
print("\n" + "=" * 130)
print("  FINAL COMPOSITE TABLE (sorted by medium composite)")
print("=" * 130)
print("    %3s  %-14s | %6s %6s %6s %6s %6s %6s %5s %6s | %6s %6s %6s %6s %6s %6s %5s %6s" % (
    "#", "variant",
    "train", "val", "test", "cov", "fc0", "fgt", "acc%", "COMP",
    "train", "val", "test", "cov", "fc0", "fgt", "acc%", "COMP"))
print("    %3s  %-14s | %55s | %55s" % ("", "", "--- LOW ---", "--- MEDIUM ---"))
print("    " + "-" * 138)

for rank, v in enumerate(scored_med, 1):
    lm = low_metrics[v]
    mm = med_metrics[v]
    lc = composite(lm)
    mc = composite(mm)
    print("    %3d  %-14s | %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %4.0f%% %6.4f | %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %4.0f%% %6.4f" % (
        rank, v,
        lm.get("train_acc") or 0, lm.get("val_acc") or 0, lm.get("test_acc") or 0,
        lm.get("cov_eq") or 0, lm.get("fc0") or 0, lm.get("fgt") or 0,
        (lm.get("accepted") or 0) * 100, lc,
        mm.get("train_acc") or 0, mm.get("val_acc") or 0, mm.get("test_acc") or 0,
        mm.get("cov_eq") or 0, mm.get("fc0") or 0, mm.get("fgt") or 0,
        (mm.get("accepted") or 0) * 100, mc))
