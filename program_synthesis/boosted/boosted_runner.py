from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_SYNTHESIS_DIR = os.path.dirname(CURRENT_DIR)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

import runner as base_runner  # noqa: E402


DEFAULT_DATASET_DIR = os.path.join("program_synthesis", "boosted", "datasets")
DEFAULT_OUTPUT_DIR = os.path.join("program_synthesis", "boosted", "outputs")

CDC_FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

CDC_NUMERIC_FEATURES = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

CDC_SEMANTIC_CONTEXT = textwrap.dedent(
    """
    Dataset: CDC diabetes indicators. The target output is binary: 1 means the person has diabetes, 0 means no diabetes.
    Inputs are dictionaries with named CDC features. Binary health/lifestyle fields use yes/no strings; Sex uses female/male.
    Numeric or ordinal fields are discretized into five qualitative bins: very low, low, medium, high, very high.
    Important fields include HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, GenHlth, DiffWalk, Age, Education, and Income.
    Write `f(x)` to accept this parsed dictionary directly, e.g. `x["BMI"] == "high"` or `x.get("HighBP") == "yes"`.
    """
).strip()


MUSHROOM_SEMANTIC_CONTEXT = textwrap.dedent(
    """
    Dataset: secondary mushroom attributes. The target output is binary: 1 means edible, 0 means poisonous.
    Inputs are dictionaries with named mushroom morphology fields such as cap_shape, cap_surface, cap_color, gill_attachment, gill_color, stem_root, stem_surface, stem_color, has_ring, ring_type, spore_print_color, habitat, and season.
    Numeric size fields cap_diameter, stem_height, and stem_width are discretized into five qualitative bins: very low, low, medium, high, very high.
    Categorical fields use readable category names such as convex, flat, smooth, sticky, brown, white, close, distant, bulbous, rooted, pendant, woods, grasses, summer, and autumn; missing values use unknown.
    Write `f(x)` to accept this parsed dictionary directly, e.g. `x["cap_shape"] == "convex"` or `x.get("stem_width") in ("high", "very high")`.
    """
).strip()


HTRU2_SEMANTIC_CONTEXT = textwrap.dedent(
    """
    Dataset: HTRU2 pulsar candidates. The target output is binary: 1 means pulsar, 0 means non-pulsar.
    Inputs are dictionaries with named numeric profile features converted to qualitative bins: very low, low, medium, high, very high.
    Pulse profile fields include profile_mean, profile_stdev, profile_skewness, and profile_kurtosis.
    Dispersion-measure signal-to-noise fields include dm_snr_mean, dm_snr_stdev, dm_snr_skewness, and dm_snr_kurtosis.
    Write `f(x)` to accept this parsed dictionary directly, e.g. `x["profile_skewness"] in ("high", "very high")`.
    """
).strip()


CHESS_SEMANTIC_CONTEXT = textwrap.dedent(
    """
    Dataset: chess King-Rook versus King-Pawn on a7 endgames. The target output is binary: 1 means white can win, 0 means no win.
    Inputs are dictionaries with UCI KRKPA7 feature abbreviations such as bkblk, bkon8, bkxwp, cntxt, dsopp, hdchk, katri, rimmx, rkxwp, simpl, skach, stlmt, thrsk, wkcti, wkna8, wknck, wkovl, and wkpos.
    Most values are true/false tactical or geometric flags. A few fields use categorical values such as black, white, none, greater, or lesser.
    Write `f(x)` to accept this parsed dictionary directly, e.g. `x.get("bkxwp") == "true"` or `x.get("wkpos") != "none"`.
    """
).strip()


MUSHROOM_HYBRID_CONTEXT = textwrap.dedent(
    """
    Dataset: secondary mushroom attributes. The target output is binary: 1 means edible, 0 means poisonous.
    Inputs are dictionaries with named mushroom morphology fields. Numeric size fields are represented twice: cap_diameter_bin, stem_height_bin, stem_width_bin use qualitative bins, and cap_diameter_z, stem_height_z, stem_width_z are numeric z-scores centered on the dataset.
    Categorical fields use compact readable strings that preserve the original code and missingness, e.g. convex|missing_no|code_x or unknown|missing_yes|code_missing.
    Prefer rules that combine readable categories with numeric z-score thresholds, e.g. x.get("cap_shape", "").startswith("convex") or float(x.get("stem_width_z", 0)) > 1.0.
    """
).strip()


HTRU2_HYBRID_CONTEXT = textwrap.dedent(
    """
    Dataset: HTRU2 pulsar candidates. The target output is binary: 1 means pulsar, 0 means non-pulsar.
    Inputs are dictionaries with named pulse-profile and dispersion-measure features. Each numeric feature has a qualitative bin field ending in _bin and a numeric z-score field ending in _z.
    Pulse profile fields include profile_mean, profile_stdev, profile_skewness, and profile_kurtosis. Dispersion-measure signal-to-noise fields include dm_snr_mean, dm_snr_stdev, dm_snr_skewness, and dm_snr_kurtosis.
    Prefer rules that threshold z-scores while using bins for readable guards, e.g. float(x.get("profile_skewness_z", 0)) > 1.0 or x.get("dm_snr_kurtosis_bin") in ("high", "very high").
    """
).strip()


CHESS_HYBRID_CONTEXT = textwrap.dedent(
    """
    Dataset: chess King-Rook versus King-Pawn on a7 endgames. The target output is binary: 1 means white can win, 0 means no win.
    Inputs are dictionaries with UCI KRKPA7 feature abbreviations. Values preserve both readable labels and original UCI codes, e.g. true|code_t, false|code_f, none|code_n, greater|code_g, lesser|code_l.
    Write rules against the readable prefix or the code token, e.g. x.get("bkxwp", "").startswith("true") or "code_t" in x.get("skach", "").
    """
).strip()


def default_cdc_representation(tabular_representation: str) -> str:
    return "semantic" if tabular_representation in {"semantic", "hybrid"} else "obfuscated"


def get_dataset_context(
    target_name: str,
    cdc_representation: str,
    tabular_representation: str = "obfuscated",
) -> Optional[str]:
    if target_name == "cdc_diabetes" and cdc_representation == "semantic":
        return CDC_SEMANTIC_CONTEXT
    if tabular_representation == "semantic":
        if target_name == "mushroom":
            return MUSHROOM_SEMANTIC_CONTEXT
        if target_name == "htru2":
            return HTRU2_SEMANTIC_CONTEXT
        if target_name == "chess":
            return CHESS_SEMANTIC_CONTEXT
    if tabular_representation == "hybrid":
        if target_name == "mushroom":
            return MUSHROOM_HYBRID_CONTEXT
        if target_name == "htru2":
            return HTRU2_HYBRID_CONTEXT
        if target_name == "chess":
            return CHESS_HYBRID_CONTEXT
        return None
    return None


def build_generation_prompt(
    data_examples: List[str],
    seq_len: int,
    decimal: bool,
    tabular: bool,
    dataset_context: Optional[str] = None,
) -> str:
    prompt = base_runner.build_user_prompt(
        data_examples,
        seq_len,
        decimal,
        tabular,
    )
    if not dataset_context:
        return prompt
    context_block = f"**Dataset Context:**\n{dataset_context}\n\n"
    marker = "**Data Examples:**"
    if marker in prompt:
        return prompt.replace(marker, context_block + marker, 1)
    return context_block + prompt


def get_model_pricing(model_name: Optional[str]) -> Optional[Dict[str, float | str]]:
    lowered = (model_name or "").strip().lower()
    if not lowered:
        return None
    if "gpt-5.2" in lowered:
        return {"name": "gpt-5.2", "input": 1.75, "output": 14.00}
    if "gpt-5.1" in lowered:
        return {"name": "gpt-5.1", "input": 1.25, "output": 10.00}
    if "gpt-5" in lowered:
        return {"name": "gpt-5", "input": 1.25, "output": 10.00}
    if "gpt-4.1" in lowered:
        return {"name": "gpt-4.1", "input": 2.00, "output": 8.00}
    if "gpt-4o" in lowered:
        return {"name": "gpt-4o", "input": 2.50, "output": 10.00}
    if "o3-mini" in lowered:
        return {"name": "o3-mini", "input": 1.10, "output": 4.40}
    if "o4-mini" in lowered:
        return {"name": "o4-mini", "input": 1.10, "output": 4.40}
    return None


def estimate_cost_usd(
    model_name: Optional[str],
    prompt_tokens: Any,
    completion_tokens: Any,
) -> Dict[str, Optional[float | str]]:
    pricing = get_model_pricing(model_name)
    if pricing is None:
        return {
            "pricing_model": None,
            "input_rate_per_million": None,
            "output_rate_per_million": None,
            "estimated_input_cost_usd": None,
            "estimated_output_cost_usd": None,
            "estimated_total_cost_usd": None,
        }
    prompt = float(prompt_tokens or 0.0)
    completion = float(completion_tokens or 0.0)
    input_cost = (prompt / 1_000_000.0) * float(pricing["input"])
    output_cost = (completion / 1_000_000.0) * float(pricing["output"])
    return {
        "pricing_model": str(pricing["name"]),
        "input_rate_per_million": float(pricing["input"]),
        "output_rate_per_million": float(pricing["output"]),
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": input_cost + output_cost,
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": int(time.time() * 1000),
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in base and k not in ("msg", "args", "levelname", "name"):
                base[k] = v
        return json.dumps(base, ensure_ascii=False)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("boosted_runner")
    logger.setLevel(level.upper())

    os.makedirs(CURRENT_DIR, exist_ok=True)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())

    file_handler = logging.FileHandler(
        os.path.join(CURRENT_DIR, "boosted_runner.log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())

    logger.handlers[:] = [stream_handler, file_handler]
    logger.propagate = False
    return logger


@dataclass
class Example:
    line: str
    x: Any
    y01: int
    ypm: int


@dataclass
class BoostConfig:
    boost_rounds: int
    batch_sizes: List[int]
    round_retries: int
    max_weak_error: float
    min_alpha: float
    num_trials: int
    stop_on_perfect_train: bool
    resample_each_retry: bool
    output_dir: str
    repair_rounds: int = 0
    repair_mistake_limit: int = 128
    repair_anchor_count: int = 16
    repair_trigger_batch_acc_below: Optional[float] = None
    repair_trigger_weighted_error_above: Optional[float] = None
    repair_trigger_min_mistakes: int = 1
    sample_without_replacement: bool = False
    strict_acceptance: bool = False
    whole_train_repair_rounds: int = 0
    whole_train_repair_batch_size: int = 256
    whole_train_mistake_frac: float = 0.7
    whole_train_recent_fix_frac: float = 0.2
    whole_train_anchor_frac: float = 0.1
    tabular_representation: str = "obfuscated"
    cdc_representation: str = "obfuscated"
    accept_best_on_failure: bool = False
    best_fallback_max_weak_error: float = 0.499


@dataclass
class BatchCandidate:
    code: str
    fn_callable: Any
    batch_acc: float
    misclassified_lines: List[str]
    correct_lines: List[str]
    eval_errors: int


@dataclass
class TrainCandidateState:
    correct_lines: List[str]
    misclassified_lines: List[str]
    train_acc: float
    weighted_error: float
    train_preds_pm: List[int]
    eval_errors: int


def _safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=os.path.dirname(path),
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _safe_write_json(path: str, obj: Any) -> None:
    _safe_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _stable_seed(*parts: Any) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def parse_examples(lines: Sequence[str], is_tabular: bool) -> List[Example]:
    examples: List[Example] = []
    for line in lines:
        x_raw, y_raw = line.split("->", 1)
        x_value = x_raw.strip()
        if is_tabular:
            x_value = base_runner._parse_tabular_input(x_value)
        y01 = int(y_raw.strip())
        examples.append(
            Example(
                line=line,
                x=x_value,
                y01=y01,
                ypm=1 if y01 == 1 else -1,
            )
        )
    return examples


def predict_01(fn_callable, x_value: Any) -> int:
    pred = fn_callable(x_value)
    return base_runner._normalize_pred_to01(pred)


def evaluate_weighted_error(
    fn_callable,
    examples: Sequence[Example],
    weights: Sequence[float],
) -> Tuple[float, List[int], int]:
    if len(examples) != len(weights):
        raise ValueError("examples and weights must have identical length.")
    if not examples:
        return 1.0, [], 0

    weighted_errors = 0.0
    predictions_pm: List[int] = []
    eval_errors = 0
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        total_weight = 1.0

    for example, weight in zip(examples, weights):
        try:
            pred01 = predict_01(fn_callable, example.x)
        except Exception:
            pred01 = 1 - example.y01
            eval_errors += 1
        pred_pm = 1 if pred01 == 1 else -1
        predictions_pm.append(pred_pm)
        if pred01 != example.y01:
            weighted_errors += weight

    return weighted_errors / total_weight, predictions_pm, eval_errors


def update_distribution(
    weights: Sequence[float],
    labels_pm: Sequence[int],
    preds_pm: Sequence[int],
    alpha: float,
) -> List[float]:
    updated = [
        float(w) * math.exp(-alpha * float(y_pm) * float(h_pm))
        for w, y_pm, h_pm in zip(weights, labels_pm, preds_pm)
    ]
    normalizer = sum(updated)
    if not math.isfinite(normalizer) or normalizer <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / normalizer for w in updated]


def sample_weighted_batch(
    examples: Sequence[Example],
    weights: Sequence[float],
    batch_size: int,
    rng: random.Random,
    without_replacement: bool = False,
) -> Tuple[List[int], List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if without_replacement:
        target_size = min(batch_size, len(examples))
        weighted_keys: List[Tuple[float, int]] = []
        for idx, weight in enumerate(weights):
            if float(weight) <= 0.0:
                continue
            u = max(rng.random(), 1e-12)
            weighted_keys.append((math.log(u) / float(weight), idx))
        weighted_keys.sort(reverse=True)
        indices = [idx for _score, idx in weighted_keys[:target_size]]
        if len(indices) < target_size:
            remaining = [idx for idx in range(len(examples)) if idx not in set(indices)]
            rng.shuffle(remaining)
            indices.extend(remaining[: target_size - len(indices)])
    else:
        indices = rng.choices(range(len(examples)), weights=weights, k=batch_size)
    return indices, [examples[idx].line for idx in indices]


def split_prediction_outcomes(
    fn_callable,
    examples: Sequence[Example],
) -> Tuple[List[str], List[str], int]:
    correct_lines: List[str] = []
    misclassified_lines: List[str] = []
    eval_errors = 0
    for example in examples:
        try:
            pred01 = predict_01(fn_callable, example.x)
        except Exception:
            pred01 = 1 - example.y01
            eval_errors += 1
        if pred01 == example.y01:
            correct_lines.append(example.line)
        else:
            misclassified_lines.append(example.line)
    return correct_lines, misclassified_lines, eval_errors


def evaluate_batch_candidate(
    code_str: str,
    batch_examples: Sequence[Example],
) -> BatchCandidate:
    fn_callable = base_runner.compile_callable_from_code(code_str)
    correct_lines, misclassified_lines, eval_errors = split_prediction_outcomes(
        fn_callable,
        batch_examples,
    )
    batch_acc = len(correct_lines) / len(batch_examples) if batch_examples else 0.0
    return BatchCandidate(
        code=code_str,
        fn_callable=fn_callable,
        batch_acc=batch_acc,
        misclassified_lines=misclassified_lines,
        correct_lines=correct_lines,
        eval_errors=eval_errors,
    )


def evaluate_train_candidate(
    fn_callable,
    train_examples: Sequence[Example],
    weights: Sequence[float],
) -> TrainCandidateState:
    correct_lines, misclassified_lines, _split_eval_errors = split_prediction_outcomes(
        fn_callable,
        train_examples,
    )
    weighted_error, train_preds_pm, eval_errors = evaluate_weighted_error(
        fn_callable,
        train_examples,
        weights,
    )
    train_acc = len(correct_lines) / len(train_examples) if train_examples else 0.0
    return TrainCandidateState(
        correct_lines=correct_lines,
        misclassified_lines=misclassified_lines,
        train_acc=train_acc,
        weighted_error=weighted_error,
        train_preds_pm=train_preds_pm,
        eval_errors=eval_errors,
    )


def _limited_lines(lines: Sequence[str], limit: int) -> List[str]:
    if limit <= 0:
        return []
    return list(lines[:limit])


def build_repair_prompt(
    current_code: str,
    mistake_lines: Sequence[str],
    anchor_lines: Sequence[str],
    seq_len: int,
    decimal: bool,
    tabular: bool,
    dataset_context: Optional[str] = None,
) -> str:
    if tabular:
        input_description = "tabular comma-separated feature:value inputs"
    elif decimal:
        input_description = f"decimal vector inputs of length {seq_len}"
    else:
        input_description = f"binary vector inputs of length {seq_len}"

    mistakes_block = "\n".join(mistake_lines) if mistake_lines else "(none)"
    anchors_block = "\n".join(anchor_lines) if anchor_lines else "(none)"
    code_block = textwrap.dedent(current_code).strip()
    lines = [
            f"Repair this Python classifier `f(x)` for {input_description}.",
            "Return exactly one valid JSON object with one key, `code`, whose value is the complete revised function.",
            "Do not include markdown, explanation, analysis, or text before or after the JSON.",
    ]
    if dataset_context:
        lines.extend(["", "Dataset context:", dataset_context])
    lines.extend(
        [
            "",
            "Current code:",
            "```",
            code_block,
            "```",
            "",
            "Wrong examples, formatted as `input -> correct_output`:",
            "```",
            mistakes_block,
            "```",
            "",
            "Correct anchor examples to preserve:",
            "```",
            anchors_block,
            "```",
            "",
            "Rules:",
            "- Keep the implementation concise and general.",
            "- Use only built-in Python.",
            "- Do not add file I/O, network calls, training loops, or lookup tables keyed by full rows.",
            "- If uncertain, make the smallest general code change that fixes the wrong examples.",
            '- Output only this JSON shape: {"code": "def f(x):\\n    ..."}',
        ]
    )
    return "\n".join(lines)


def build_minimal_repair_prompt(
    current_code: str,
    mistake_lines: Sequence[str],
    anchor_lines: Sequence[str],
    dataset_context: Optional[str] = None,
) -> str:
    mistakes_block = "\n".join(mistake_lines) if mistake_lines else "(none)"
    anchors_block = "\n".join(anchor_lines) if anchor_lines else "(none)"
    code_block = textwrap.dedent(current_code).strip()
    lines = [
            "Return only valid JSON. No markdown. No explanation.",
            'JSON schema: {"code": "def f(x):\\n    ..."}',
    ]
    if dataset_context:
        lines.extend(["", "Dataset context:", dataset_context])
    lines.extend(
        [
            "",
            "Revise this function:",
            "```",
            code_block,
            "```",
            "",
            "It is wrong on:",
            "```",
            mistakes_block,
            "```",
            "",
            "Keep these correct if possible:",
            "```",
            anchors_block,
            "```",
        ]
    )
    return "\n".join(lines)


def build_whole_train_repair_prompt(
    current_code: str,
    mistake_lines: Sequence[str],
    recently_fixed_lines: Sequence[str],
    anchor_lines: Sequence[str],
    seq_len: int,
    decimal: bool,
    tabular: bool,
    dataset_context: Optional[str] = None,
) -> str:
    if tabular:
        input_description = "tabular comma-separated feature:value inputs"
    elif decimal:
        input_description = f"decimal vector inputs of length {seq_len}"
    else:
        input_description = f"binary vector inputs of length {seq_len}"

    mistakes_block = "\n".join(mistake_lines) if mistake_lines else "(none)"
    recent_block = "\n".join(recently_fixed_lines) if recently_fixed_lines else "(none)"
    anchors_block = "\n".join(anchor_lines) if anchor_lines else "(none)"
    code_block = textwrap.dedent(current_code).strip()
    lines = [
            f"Repair this Python classifier `f(x)` for {input_description}.",
            "You are in an iterative curriculum repair loop over the whole training set.",
            "Return exactly one valid JSON object with one key, `code`, whose value is the complete revised function.",
            "Do not include markdown, explanation, analysis, or text before or after the JSON.",
    ]
    if dataset_context:
        lines.extend(["", "Dataset context:", dataset_context])
    lines.extend(
        [
            "",
            "Current code:",
            "```",
            code_block,
            "```",
            "",
            "Wrong examples that still need to be fixed:",
            "```",
            mistakes_block,
            "```",
            "",
            "Recently fixed examples that should stay fixed:",
            "```",
            recent_block,
            "```",
            "",
            "Correct anchor examples to preserve:",
            "```",
            anchors_block,
            "```",
            "",
            "Rules:",
            "- Keep the implementation concise and general.",
            "- Use only built-in Python.",
            "- Do not add file I/O, network calls, training loops, or lookup tables keyed by full rows.",
            "- Prefer small general rule changes over memorizing examples.",
            '- Output only this JSON shape: {"code": "def f(x):\\n    ..."}',
        ]
    )
    return "\n".join(lines)


def _sample_unique_lines(lines: Sequence[str], count: int, rng: random.Random) -> List[str]:
    if count <= 0 or not lines:
        return []
    if count >= len(lines):
        return list(lines)
    return rng.sample(list(lines), count)


def compose_whole_train_repair_batch(
    mistake_lines: Sequence[str],
    recently_fixed_lines: Sequence[str],
    anchor_lines: Sequence[str],
    total_size: int,
    mistake_frac: float,
    recent_fix_frac: float,
    anchor_frac: float,
    rng: random.Random,
) -> Dict[str, List[str]]:
    if total_size <= 0:
        return {"mistakes": [], "recently_fixed": [], "anchors": [], "combined": []}

    target_recent = max(0, int(round(total_size * recent_fix_frac)))
    target_anchor = max(0, int(round(total_size * anchor_frac)))
    target_mistakes = max(0, total_size - target_recent - target_anchor)

    selected_mistakes = _sample_unique_lines(mistake_lines, target_mistakes, rng)
    selected_recent = _sample_unique_lines(recently_fixed_lines, target_recent, rng)
    selected_anchors = _sample_unique_lines(anchor_lines, target_anchor, rng)

    combined = list(selected_mistakes) + list(selected_recent) + list(selected_anchors)
    selected_set = set(combined)
    remaining_pools = [
        [line for line in mistake_lines if line not in selected_set],
        [line for line in recently_fixed_lines if line not in selected_set],
        [line for line in anchor_lines if line not in selected_set],
    ]
    while len(combined) < total_size:
        filled = False
        for pool in remaining_pools:
            if not pool:
                continue
            pick_idx = rng.randrange(len(pool))
            picked = pool.pop(pick_idx)
            if picked in selected_set:
                continue
            combined.append(picked)
            selected_set.add(picked)
            filled = True
            if len(combined) >= total_size:
                break
        if not filled:
            break

    rng.shuffle(combined)
    return {
        "mistakes": selected_mistakes,
        "recently_fixed": selected_recent,
        "anchors": selected_anchors,
        "combined": combined,
    }


def should_run_batch_repair(
    candidate: BatchCandidate,
    cfg: BoostConfig,
    initial_weighted_error: Optional[float],
) -> Tuple[bool, str]:
    if cfg.repair_rounds <= 0:
        return False, "repair_disabled"
    if len(candidate.misclassified_lines) < cfg.repair_trigger_min_mistakes:
        return False, "skipped_gate_min_mistakes"

    batch_gate = cfg.repair_trigger_batch_acc_below
    weighted_gate = cfg.repair_trigger_weighted_error_above
    if batch_gate is None and weighted_gate is None:
        return True, "run_default"

    trigger_reasons: List[str] = []
    if batch_gate is not None and candidate.batch_acc < batch_gate:
        trigger_reasons.append("batch_acc")
    if weighted_gate is not None and initial_weighted_error is not None and initial_weighted_error > weighted_gate:
        trigger_reasons.append("weighted_error")
    if trigger_reasons:
        return True, "run_gate_" + "_".join(trigger_reasons)

    if batch_gate is not None and candidate.batch_acc >= batch_gate:
        return False, "skipped_gate_batch_acc"
    if weighted_gate is not None:
        return False, "skipped_gate_weighted_error"
    return False, "skipped_gate"


def should_accept_whole_train_repair(
    current_state: TrainCandidateState,
    candidate_state: TrainCandidateState,
) -> bool:
    eps = 1e-12
    if candidate_state.train_acc > current_state.train_acc + eps:
        return True
    if candidate_state.train_acc < current_state.train_acc - eps:
        return False
    if candidate_state.weighted_error < current_state.weighted_error - eps:
        return True
    if candidate_state.weighted_error > current_state.weighted_error + eps:
        return False
    return len(candidate_state.misclassified_lines) < len(current_state.misclassified_lines)


def evaluate_ensemble_accuracy(
    learners: Sequence[Dict[str, Any]],
    examples: Sequence[Example],
) -> float:
    if not learners or not examples:
        return 0.0
    correct = 0
    for example in examples:
        score = 0.0
        for learner in learners:
            try:
                pred01 = predict_01(learner["callable"], example.x)
            except Exception:
                pred01 = 0
            score += learner["alpha"] * (1.0 if pred01 == 1 else -1.0)
        pred01 = 1 if score >= 0.0 else 0
        correct += int(pred01 == example.y01)
    return correct / len(examples)


def extract_function_name(code_str: str) -> str:
    tree = ast.parse(textwrap.dedent(code_str.strip()))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("No function definition found in learner code.")


def build_ensemble_module(learners: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = [
        '"""Auto-generated boosted ensemble wrapper."""',
        "from typing import Any",
        "",
    ]

    for idx, learner in enumerate(learners, start=1):
        code = textwrap.dedent(learner["code"]).strip()
        fn_name = extract_function_name(code)
        parts.append(f"# Learner {idx}")
        parts.append(code)
        parts.append(f"h_{idx} = {fn_name}")
        parts.append("")

    alpha_str = ", ".join(f"{learner['alpha']:.16g}" for learner in learners)
    learner_refs = ", ".join(f"h_{idx}" for idx in range(1, len(learners) + 1))
    parts.extend(
        [
            f"ALPHAS = [{alpha_str}]",
            f"LEARNERS = [{learner_refs}]",
            "",
            "def _normalize_pred_to_pm1(pred: Any) -> int:",
            "    try:",
            "        if hasattr(pred, 'item'):",
            "            pred = pred.item()",
            "    except Exception:",
            "        pass",
            "    if isinstance(pred, bool):",
            "        return 1 if pred else -1",
            "    if isinstance(pred, int):",
            "        return 1 if pred != 0 else -1",
            "    if isinstance(pred, str):",
            "        s = pred.strip().strip(\"\\\"'\")",
            "        if s in ('1', 'true', 'True'):",
            "            return 1",
            "        if s in ('0', 'false', 'False', ''):",
            "            return -1",
            "        try:",
            "            return 1 if int(float(s)) != 0 else -1",
            "        except Exception:",
            "            return 1 if s else -1",
            "    return 1 if pred else -1",
            "",
            "def f(x: Any) -> int:",
            "    score = 0.0",
            "    for alpha, learner in zip(ALPHAS, LEARNERS):",
            "        score += alpha * _normalize_pred_to_pm1(learner(x))",
            "    return 1 if score >= 0.0 else 0",
            "",
        ]
    )
    return "\n".join(parts)


def serialize_attempt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    serializable = dict(row)
    serializable.pop("callable", None)
    return serializable


def summarize_response_accounting(
    res: Dict[str, Any],
    fallback_model: Optional[str],
) -> Dict[str, Any]:
    usage = res.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = usage.get("reasoning_tokens")
    returned_model = res.get("returned_model")
    request_model = res.get("request_model") or fallback_model
    pricing_model_name = returned_model or request_model or fallback_model
    cost_info = estimate_cost_usd(pricing_model_name, prompt_tokens, completion_tokens)
    return {
        "request_model": request_model,
        "returned_model": returned_model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "pricing_model": cost_info["pricing_model"],
        "input_rate_per_million": cost_info["input_rate_per_million"],
        "output_rate_per_million": cost_info["output_rate_per_million"],
        "estimated_input_cost_usd": float(cost_info["estimated_input_cost_usd"] or 0.0),
        "estimated_output_cost_usd": float(cost_info["estimated_output_cost_usd"] or 0.0),
        "estimated_total_cost_usd": float(cost_info["estimated_total_cost_usd"] or 0.0),
        "tool_uses": int(res.get("tool_uses") or 0),
        "duration_ms": int(res.get("duration_ms") or 0),
    }


async def run_boosting_trial(
    client: base_runner.Runner,
    log: logging.Logger,
    fn: str,
    length: int,
    target_name: str,
    train_lines: Sequence[str],
    val_lines: Sequence[str],
    test_lines: Sequence[str],
    is_decimal: bool,
    is_tabular: bool,
    cfg: BoostConfig,
    batch_size: int,
    trial_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_examples = parse_examples(train_lines, is_tabular)
    val_examples = parse_examples(val_lines, is_tabular)
    test_examples = parse_examples(test_lines, is_tabular)
    if not train_examples:
        raise ValueError("Training split is empty.")

    weights = [1.0 / len(train_examples)] * len(train_examples)
    learners: List[Dict[str, Any]] = []
    accepted_rounds: List[Dict[str, Any]] = []
    attempt_rows: List[Dict[str, Any]] = []
    labels_pm = [example.ypm for example in train_examples]
    rng = random.Random(_stable_seed("boost", fn, length, batch_size, trial_idx, client.cfg.seed))
    attempt_counter = 0
    stopped_reason = "max_rounds_reached"
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_estimated_cost_usd = 0.0
    fallback_model = getattr(getattr(client, "cfg", None), "model", None)
    dataset_context = get_dataset_context(target_name, cfg.cdc_representation, cfg.tabular_representation)

    def add_response_accounting(res: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal total_prompt_tokens
        nonlocal total_completion_tokens
        nonlocal total_reasoning_tokens
        nonlocal total_estimated_cost_usd
        accounting = summarize_response_accounting(res, fallback_model)
        total_prompt_tokens += int(accounting["prompt_tokens"] or 0)
        total_completion_tokens += int(accounting["completion_tokens"] or 0)
        total_reasoning_tokens += int(accounting["reasoning_tokens"] or 0)
        total_estimated_cost_usd += float(accounting["estimated_total_cost_usd"] or 0.0)
        return accounting

    for round_idx in range(1, cfg.boost_rounds + 1):
        sampled_indices, sampled_lines = sample_weighted_batch(
            train_examples,
            weights,
            batch_size,
            rng,
            without_replacement=cfg.sample_without_replacement,
        )
        sampled_unique = len(set(sampled_indices))
        accepted_this_round = False
        best_fallback: Optional[Dict[str, Any]] = None

        for retry_idx in range(1, cfg.round_retries + 1):
            if retry_idx > 1 and cfg.resample_each_retry:
                sampled_indices, sampled_lines = sample_weighted_batch(
                    train_examples,
                    weights,
                    batch_size,
                    rng,
                    without_replacement=cfg.sample_without_replacement,
                )
                sampled_unique = len(set(sampled_indices))

            api_attempt_start = attempt_counter + 1
            attempt_counter += 1
            if dataset_context:
                initial_prompt = build_generation_prompt(
                    list(sampled_lines),
                    length,
                    is_decimal,
                    is_tabular,
                    dataset_context=dataset_context,
                )
                res = await client._call_once(
                    fn,
                    length,
                    attempt_counter,
                    sampled_lines,
                    decimal=is_decimal,
                    tabular=is_tabular,
                    prompt_override=initial_prompt,
                )
            else:
                res = await client._call_once(
                    fn,
                    length,
                    attempt_counter,
                    sampled_lines,
                    decimal=is_decimal,
                    tabular=is_tabular,
                )

            code_str = base_runner.extract_code_from_output(res.get("text") or "")
            first_accounting = add_response_accounting(res)
            chain_prompt_tokens = int(first_accounting["prompt_tokens"] or 0)
            chain_completion_tokens = int(first_accounting["completion_tokens"] or 0)
            chain_reasoning_tokens = int(first_accounting["reasoning_tokens"] or 0)
            chain_input_cost = float(first_accounting["estimated_input_cost_usd"] or 0.0)
            chain_output_cost = float(first_accounting["estimated_output_cost_usd"] or 0.0)
            chain_total_cost = float(first_accounting["estimated_total_cost_usd"] or 0.0)
            chain_tool_uses = int(first_accounting["tool_uses"] or 0)
            chain_duration_ms = int(first_accounting["duration_ms"] or 0)
            request_model = first_accounting["request_model"]
            returned_model = first_accounting["returned_model"]
            pricing_model = first_accounting["pricing_model"]
            input_rate_per_million = first_accounting["input_rate_per_million"]
            output_rate_per_million = first_accounting["output_rate_per_million"]
            batch_examples = [train_examples[idx] for idx in sampled_indices]
            repair_history: List[Dict[str, Any]] = []
            row: Dict[str, Any] = {
                "fn": fn,
                "target_name": target_name,
                "length": length,
                "batch_size": batch_size,
                "trial": trial_idx,
                "round": round_idx,
                "retry": retry_idx,
                "attempt": api_attempt_start,
                "api_attempt_start": api_attempt_start,
                "api_attempt_end": attempt_counter,
                "accepted": False,
                "sampled_unique": sampled_unique,
                "sampled_duplicate_count": len(sampled_indices) - sampled_unique,
                "compile_error": None,
                "weighted_error": None,
                "alpha": None,
                "batch_acc": None,
                "train_acc": None,
                "val_acc": None,
                "test_acc": None,
                "ensemble_train_acc": None,
                "ensemble_val_acc": None,
                "ensemble_test_acc": None,
                "request_model": request_model,
                "returned_model": returned_model,
                "prompt_tokens": chain_prompt_tokens,
                "completion_tokens": chain_completion_tokens,
                "reasoning_tokens": chain_reasoning_tokens,
                "pricing_model": pricing_model,
                "input_rate_per_million": input_rate_per_million,
                "output_rate_per_million": output_rate_per_million,
                "estimated_input_cost_usd": chain_input_cost,
                "estimated_output_cost_usd": chain_output_cost,
                "estimated_total_cost_usd": chain_total_cost,
                "tool_uses": chain_tool_uses,
                "duration_ms": chain_duration_ms,
                "repair_rounds_requested": cfg.repair_rounds,
                "repair_calls": 0,
                "repair_best_step": None,
                "repair_gate_reason": None,
                "repair_gate_weighted_error": None,
                "repair_initial_batch_acc": None,
                "repair_final_batch_acc": None,
                "repair_history": None,
                "whole_train_repair_rounds_requested": cfg.whole_train_repair_rounds,
            }

            if not code_str:
                row["compile_error"] = "no_code_found"
                attempt_rows.append(serialize_attempt_row(row))
                continue

            try:
                best_candidate = evaluate_batch_candidate(code_str, batch_examples)
                repair_history.append(
                    {
                        "step": 0,
                        "status": "initial",
                        "batch_acc": best_candidate.batch_acc,
                        "mistakes": len(best_candidate.misclassified_lines),
                        "eval_errors": best_candidate.eval_errors,
                        "accepted_as_best": True,
                    }
                )
                initial_train_state: Optional[TrainCandidateState] = None
                initial_weighted_error_for_gate: Optional[float] = None
                if cfg.repair_trigger_weighted_error_above is not None:
                    initial_train_state = evaluate_train_candidate(best_candidate.fn_callable, train_examples, weights)
                    initial_weighted_error_for_gate = initial_train_state.weighted_error

                should_repair, repair_gate_reason = should_run_batch_repair(
                    best_candidate,
                    cfg,
                    initial_weighted_error_for_gate,
                )
                row["repair_gate_reason"] = repair_gate_reason
                row["repair_gate_weighted_error"] = initial_weighted_error_for_gate

                if should_repair:
                    for repair_step in range(1, cfg.repair_rounds + 1):
                        if not best_candidate.misclassified_lines:
                            repair_history.append(
                                {
                                    "step": repair_step,
                                    "status": "skipped_no_mistakes",
                                    "batch_acc": best_candidate.batch_acc,
                                    "mistakes": 0,
                                    "accepted_as_best": False,
                                }
                            )
                            break

                        mistake_lines = _limited_lines(
                            best_candidate.misclassified_lines,
                            cfg.repair_mistake_limit,
                        )
                        anchor_lines = _limited_lines(
                            best_candidate.correct_lines,
                            cfg.repair_anchor_count,
                        )
                        repair_prompt = build_repair_prompt(
                            best_candidate.code,
                            mistake_lines,
                            anchor_lines,
                            length,
                            is_decimal,
                            is_tabular,
                            dataset_context=dataset_context,
                        )

                        attempt_counter += 1
                        repair_res = await client._call_once(
                            fn,
                            length,
                            attempt_counter,
                            mistake_lines,
                            decimal=is_decimal,
                            tabular=is_tabular,
                            prompt_override=repair_prompt,
                        )
                        repair_accounting = add_response_accounting(repair_res)
                        chain_prompt_tokens += int(repair_accounting["prompt_tokens"] or 0)
                        chain_completion_tokens += int(repair_accounting["completion_tokens"] or 0)
                        chain_reasoning_tokens += int(repair_accounting["reasoning_tokens"] or 0)
                        chain_input_cost += float(repair_accounting["estimated_input_cost_usd"] or 0.0)
                        chain_output_cost += float(repair_accounting["estimated_output_cost_usd"] or 0.0)
                        chain_total_cost += float(repair_accounting["estimated_total_cost_usd"] or 0.0)
                        chain_tool_uses += int(repair_accounting["tool_uses"] or 0)
                        chain_duration_ms += int(repair_accounting["duration_ms"] or 0)
                        request_model = repair_accounting["request_model"] or request_model
                        returned_model = repair_accounting["returned_model"] or returned_model
                        pricing_model = repair_accounting["pricing_model"] or pricing_model
                        input_rate_per_million = repair_accounting["input_rate_per_million"] or input_rate_per_million
                        output_rate_per_million = repair_accounting["output_rate_per_million"] or output_rate_per_million

                        repair_text = repair_res.get("text") or ""
                        repaired_code = base_runner.extract_code_from_output(repair_text)
                        repair_status = "repair"
                        if not repaired_code:
                            repair_history.append(
                                {
                                    "step": repair_step,
                                    "status": "no_code_found",
                                    "response_chars": len(repair_text),
                                    "empty_response": len(repair_text.strip()) == 0,
                                    "batch_acc": best_candidate.batch_acc,
                                    "mistakes": len(best_candidate.misclassified_lines),
                                    "accepted_as_best": False,
                                }
                            )
                            fallback_prompt = build_minimal_repair_prompt(
                                best_candidate.code,
                                _limited_lines(mistake_lines, min(len(mistake_lines), 8)),
                                _limited_lines(anchor_lines, min(len(anchor_lines), 4)),
                                dataset_context=dataset_context,
                            )
                            attempt_counter += 1
                            fallback_res = await client._call_once(
                                fn,
                                length,
                                attempt_counter,
                                mistake_lines,
                                decimal=is_decimal,
                                tabular=is_tabular,
                                prompt_override=fallback_prompt,
                            )
                            fallback_accounting = add_response_accounting(fallback_res)
                            chain_prompt_tokens += int(fallback_accounting["prompt_tokens"] or 0)
                            chain_completion_tokens += int(fallback_accounting["completion_tokens"] or 0)
                            chain_reasoning_tokens += int(fallback_accounting["reasoning_tokens"] or 0)
                            chain_input_cost += float(fallback_accounting["estimated_input_cost_usd"] or 0.0)
                            chain_output_cost += float(fallback_accounting["estimated_output_cost_usd"] or 0.0)
                            chain_total_cost += float(fallback_accounting["estimated_total_cost_usd"] or 0.0)
                            chain_tool_uses += int(fallback_accounting["tool_uses"] or 0)
                            chain_duration_ms += int(fallback_accounting["duration_ms"] or 0)
                            request_model = fallback_accounting["request_model"] or request_model
                            returned_model = fallback_accounting["returned_model"] or returned_model
                            pricing_model = fallback_accounting["pricing_model"] or pricing_model
                            input_rate_per_million = fallback_accounting["input_rate_per_million"] or input_rate_per_million
                            output_rate_per_million = fallback_accounting["output_rate_per_million"] or output_rate_per_million

                            fallback_text = fallback_res.get("text") or ""
                            repaired_code = base_runner.extract_code_from_output(fallback_text)
                            repair_status = "fallback_repair"
                            if not repaired_code:
                                repair_history.append(
                                    {
                                        "step": repair_step,
                                        "status": "fallback_no_code_found",
                                        "response_chars": len(fallback_text),
                                        "empty_response": len(fallback_text.strip()) == 0,
                                        "batch_acc": best_candidate.batch_acc,
                                        "mistakes": len(best_candidate.misclassified_lines),
                                        "accepted_as_best": False,
                                    }
                                )
                                break

                        try:
                            repaired_candidate = evaluate_batch_candidate(repaired_code, batch_examples)
                        except Exception as repair_exc:
                            repair_history.append(
                                {
                                    "step": repair_step,
                                    "status": f"{repair_status}_compile_error",
                                    "error": str(repair_exc),
                                    "batch_acc": best_candidate.batch_acc,
                                    "mistakes": len(best_candidate.misclassified_lines),
                                    "accepted_as_best": False,
                                }
                            )
                            break

                        accepted_repair = repaired_candidate.batch_acc >= best_candidate.batch_acc
                        if accepted_repair:
                            best_candidate = repaired_candidate
                        repair_history.append(
                            {
                                "step": repair_step,
                                "status": repair_status,
                                "batch_acc": repaired_candidate.batch_acc,
                                "mistakes": len(repaired_candidate.misclassified_lines),
                                "eval_errors": repaired_candidate.eval_errors,
                                "accepted_as_best": accepted_repair,
                            }
                        )
                else:
                    repair_history.append(
                        {
                            "step": 1,
                            "status": repair_gate_reason,
                            "batch_acc": best_candidate.batch_acc,
                            "mistakes": len(best_candidate.misclassified_lines),
                            "accepted_as_best": False,
                        }
                    )

                code_str = best_candidate.code
                fn_callable = best_candidate.fn_callable
                batch_acc = best_candidate.batch_acc
                train_state = evaluate_train_candidate(fn_callable, train_examples, weights)
                current_recently_fixed_lines: List[str] = []
                previous_mistake_lines = list(train_state.misclassified_lines)

                for repair_step in range(1, cfg.whole_train_repair_rounds + 1):
                    if not train_state.misclassified_lines:
                        repair_history.append(
                            {
                                "step": cfg.repair_rounds + repair_step,
                                "status": "whole_train_skipped_no_mistakes",
                                "train_acc": train_state.train_acc,
                                "mistakes": 0,
                                "accepted_as_best": False,
                            }
                        )
                        break

                    sampled_curriculum = compose_whole_train_repair_batch(
                        train_state.misclassified_lines,
                        current_recently_fixed_lines,
                        train_state.correct_lines,
                        cfg.whole_train_repair_batch_size,
                        cfg.whole_train_mistake_frac,
                        cfg.whole_train_recent_fix_frac,
                        cfg.whole_train_anchor_frac,
                        rng,
                    )
                    whole_prompt = build_whole_train_repair_prompt(
                        code_str,
                        sampled_curriculum["mistakes"],
                        sampled_curriculum["recently_fixed"],
                        sampled_curriculum["anchors"],
                        length,
                        is_decimal,
                        is_tabular,
                        dataset_context=dataset_context,
                    )

                    attempt_counter += 1
                    whole_res = await client._call_once(
                        fn,
                        length,
                        attempt_counter,
                        sampled_curriculum["combined"],
                        decimal=is_decimal,
                        tabular=is_tabular,
                        prompt_override=whole_prompt,
                    )
                    whole_accounting = add_response_accounting(whole_res)
                    chain_prompt_tokens += int(whole_accounting["prompt_tokens"] or 0)
                    chain_completion_tokens += int(whole_accounting["completion_tokens"] or 0)
                    chain_reasoning_tokens += int(whole_accounting["reasoning_tokens"] or 0)
                    chain_input_cost += float(whole_accounting["estimated_input_cost_usd"] or 0.0)
                    chain_output_cost += float(whole_accounting["estimated_output_cost_usd"] or 0.0)
                    chain_total_cost += float(whole_accounting["estimated_total_cost_usd"] or 0.0)
                    chain_tool_uses += int(whole_accounting["tool_uses"] or 0)
                    chain_duration_ms += int(whole_accounting["duration_ms"] or 0)
                    request_model = whole_accounting["request_model"] or request_model
                    returned_model = whole_accounting["returned_model"] or returned_model
                    pricing_model = whole_accounting["pricing_model"] or pricing_model
                    input_rate_per_million = whole_accounting["input_rate_per_million"] or input_rate_per_million
                    output_rate_per_million = whole_accounting["output_rate_per_million"] or output_rate_per_million

                    whole_text = whole_res.get("text") or ""
                    whole_code = base_runner.extract_code_from_output(whole_text)
                    history_step = cfg.repair_rounds + repair_step
                    if not whole_code:
                        repair_history.append(
                            {
                                "step": history_step,
                                "status": "whole_train_no_code_found",
                                "response_chars": len(whole_text),
                                "empty_response": len(whole_text.strip()) == 0,
                                "train_acc": train_state.train_acc,
                                "mistakes": len(train_state.misclassified_lines),
                                "sampled_mistakes": len(sampled_curriculum["mistakes"]),
                                "sampled_recently_fixed": len(sampled_curriculum["recently_fixed"]),
                                "sampled_anchors": len(sampled_curriculum["anchors"]),
                                "accepted_as_best": False,
                            }
                        )
                        break

                    try:
                        whole_candidate_callable = base_runner.compile_callable_from_code(whole_code)
                        candidate_train_state = evaluate_train_candidate(
                            whole_candidate_callable,
                            train_examples,
                            weights,
                        )
                    except Exception as whole_exc:
                        repair_history.append(
                            {
                                "step": history_step,
                                "status": "whole_train_compile_error",
                                "error": str(whole_exc),
                                "train_acc": train_state.train_acc,
                                "mistakes": len(train_state.misclassified_lines),
                                "accepted_as_best": False,
                            }
                        )
                        break

                    accepted_whole_train = should_accept_whole_train_repair(train_state, candidate_train_state)
                    if accepted_whole_train:
                        code_str = whole_code
                        fn_callable = whole_candidate_callable
                        train_state = candidate_train_state
                        current_recently_fixed_lines = [
                            line for line in previous_mistake_lines if line not in set(train_state.misclassified_lines)
                        ]
                        previous_mistake_lines = list(train_state.misclassified_lines)

                    repair_history.append(
                        {
                            "step": history_step,
                            "status": "whole_train_repair",
                            "train_acc": candidate_train_state.train_acc,
                            "weighted_error": candidate_train_state.weighted_error,
                            "mistakes": len(candidate_train_state.misclassified_lines),
                            "sampled_mistakes": len(sampled_curriculum["mistakes"]),
                            "sampled_recently_fixed": len(sampled_curriculum["recently_fixed"]),
                            "sampled_anchors": len(sampled_curriculum["anchors"]),
                            "accepted_as_best": accepted_whole_train,
                        }
                    )

                batch_acc = evaluate_batch_candidate(code_str, batch_examples).batch_acc
                train_acc = train_state.train_acc
                val_acc = (
                    base_runner.evaluate_accuracy(fn_callable, list(val_lines), log, is_tabular)
                    if val_lines
                    else None
                )
                test_acc = base_runner.evaluate_accuracy(fn_callable, list(test_lines), log, is_tabular)
                weighted_error = train_state.weighted_error
                train_preds_pm = train_state.train_preds_pm
                eval_errors = train_state.eval_errors

                eps = min(max(weighted_error, 1e-12), 1.0 - 1e-12)
                alpha = 0.5 * math.log((1.0 - eps) / eps)

                row.update(
                    {
                        "weighted_error": weighted_error,
                        "alpha": alpha,
                        "batch_acc": batch_acc,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "eval_errors": eval_errors,
                        "api_attempt_end": attempt_counter,
                        "request_model": request_model,
                        "returned_model": returned_model,
                        "prompt_tokens": chain_prompt_tokens,
                        "completion_tokens": chain_completion_tokens,
                        "reasoning_tokens": chain_reasoning_tokens,
                        "pricing_model": pricing_model,
                        "input_rate_per_million": input_rate_per_million,
                        "output_rate_per_million": output_rate_per_million,
                        "estimated_input_cost_usd": chain_input_cost,
                        "estimated_output_cost_usd": chain_output_cost,
                        "estimated_total_cost_usd": chain_total_cost,
                        "tool_uses": chain_tool_uses,
                        "duration_ms": chain_duration_ms,
                        "repair_calls": max(0, attempt_counter - api_attempt_start),
                        "repair_best_step": next(
                            (
                                item["step"]
                                for item in reversed(repair_history)
                                if item.get("accepted_as_best")
                            ),
                            0,
                        ),
                        "repair_initial_batch_acc": repair_history[0]["batch_acc"] if repair_history else batch_acc,
                        "repair_final_batch_acc": batch_acc,
                        "repair_history": json.dumps(repair_history, ensure_ascii=False),
                    }
                )

                if weighted_error <= cfg.max_weak_error and alpha >= cfg.min_alpha:
                    weights = update_distribution(weights, labels_pm, train_preds_pm, alpha)
                    learner = {
                        "round": round_idx,
                        "retry": retry_idx,
                        "alpha": alpha,
                        "weighted_error": weighted_error,
                        "code": code_str,
                        "callable": fn_callable,
                        "batch_acc": batch_acc,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                    }
                    learners.append(learner)
                    accepted_rounds.append(
                        {
                            k: v for k, v in learner.items() if k != "callable"
                        }
                    )

                    row["accepted"] = True
                    row["ensemble_train_acc"] = evaluate_ensemble_accuracy(learners, train_examples)
                    row["ensemble_val_acc"] = (
                        evaluate_ensemble_accuracy(learners, val_examples) if val_examples else None
                    )
                    row["ensemble_test_acc"] = evaluate_ensemble_accuracy(learners, test_examples)
                    attempt_rows.append(serialize_attempt_row(row))
                    accepted_this_round = True

                    if cfg.stop_on_perfect_train and row["ensemble_train_acc"] >= 1.0:
                        stopped_reason = "perfect_train_acc"
                    break

                serialized_row = serialize_attempt_row(row)
                if (
                    cfg.accept_best_on_failure
                    and weighted_error <= cfg.best_fallback_max_weak_error
                    and alpha >= cfg.min_alpha
                    and (
                        best_fallback is None
                        or weighted_error < float(best_fallback["weighted_error"])
                    )
                ):
                    best_fallback = {
                        "attempt_row_index": len(attempt_rows),
                        "row": serialized_row,
                        "round": round_idx,
                        "retry": retry_idx,
                        "alpha": alpha,
                        "weighted_error": weighted_error,
                        "code": code_str,
                        "fn_callable": fn_callable,
                        "batch_acc": batch_acc,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "train_preds_pm": train_preds_pm,
                    }
                attempt_rows.append(serialized_row)
            except Exception as exc:
                row["compile_error"] = str(exc)
                attempt_rows.append(serialize_attempt_row(row))

        if not accepted_this_round:
            if best_fallback is not None:
                weights = update_distribution(
                    weights,
                    labels_pm,
                    best_fallback["train_preds_pm"],
                    float(best_fallback["alpha"]),
                )
                learner = {
                    "round": best_fallback["round"],
                    "retry": best_fallback["retry"],
                    "alpha": best_fallback["alpha"],
                    "weighted_error": best_fallback["weighted_error"],
                    "code": best_fallback["code"],
                    "callable": best_fallback["fn_callable"],
                    "batch_acc": best_fallback["batch_acc"],
                    "train_acc": best_fallback["train_acc"],
                    "val_acc": best_fallback["val_acc"],
                    "test_acc": best_fallback["test_acc"],
                    "accepted_best_on_failure": True,
                }
                learners.append(learner)
                accepted_rounds.append({k: v for k, v in learner.items() if k != "callable"})
                row_idx = int(best_fallback["attempt_row_index"])
                attempt_rows[row_idx]["accepted"] = True
                attempt_rows[row_idx]["accepted_best_on_failure"] = True
                attempt_rows[row_idx]["ensemble_train_acc"] = evaluate_ensemble_accuracy(learners, train_examples)
                attempt_rows[row_idx]["ensemble_val_acc"] = (
                    evaluate_ensemble_accuracy(learners, val_examples) if val_examples else None
                )
                attempt_rows[row_idx]["ensemble_test_acc"] = evaluate_ensemble_accuracy(learners, test_examples)
                accepted_this_round = True
                stopped_reason = "max_rounds_reached"
                if cfg.stop_on_perfect_train and attempt_rows[row_idx]["ensemble_train_acc"] >= 1.0:
                    stopped_reason = "perfect_train_acc"
            else:
                stopped_reason = "no_acceptable_weak_learner"
                break
        if stopped_reason == "perfect_train_acc":
            break

    final_train_acc = evaluate_ensemble_accuracy(learners, train_examples)
    final_val_acc = evaluate_ensemble_accuracy(learners, val_examples) if val_examples else None
    final_test_acc = evaluate_ensemble_accuracy(learners, test_examples)

    summary = {
        "fn": fn,
        "target_name": target_name,
        "length": length,
        "batch_size": batch_size,
        "trial": trial_idx,
        "accepted_rounds": len(learners),
        "requested_rounds": cfg.boost_rounds,
        "repair_rounds": cfg.repair_rounds,
        "repair_mistake_limit": cfg.repair_mistake_limit,
        "repair_anchor_count": cfg.repair_anchor_count,
        "repair_trigger_batch_acc_below": cfg.repair_trigger_batch_acc_below,
        "repair_trigger_weighted_error_above": cfg.repair_trigger_weighted_error_above,
        "repair_trigger_min_mistakes": cfg.repair_trigger_min_mistakes,
        "whole_train_repair_rounds": cfg.whole_train_repair_rounds,
        "whole_train_repair_batch_size": cfg.whole_train_repair_batch_size,
        "sample_without_replacement": cfg.sample_without_replacement,
        "strict_acceptance": cfg.strict_acceptance,
        "tabular_representation": cfg.tabular_representation,
        "cdc_representation": cfg.cdc_representation,
        "accept_best_on_failure": cfg.accept_best_on_failure,
        "best_fallback_max_weak_error": cfg.best_fallback_max_weak_error,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "api_attempt_count": attempt_counter,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_estimated_cost_usd": total_estimated_cost_usd,
        "stopped_reason": stopped_reason,
    }
    return summary, attempt_rows, accepted_rounds


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        _safe_write_text(path, "")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=os.path.dirname(path),
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if body:
        body += "\n"
    _safe_write_text(path, body)


def build_base_config(args: argparse.Namespace) -> base_runner.Config:
    cfg = base_runner.Config()
    cfg.functions = list(args.functions)
    cfg.lengths = list(args.lengths)
    cfg.lengths_explicit = True
    cfg.train_size = args.train_size
    cfg.val_size = args.val_size
    cfg.test_size = args.test_size
    cfg.seed = args.seed
    tabular_representation = getattr(args, "tabular_representation", "obfuscated")
    cdc_representation = getattr(args, "cdc_representation", None) or default_cdc_representation(tabular_representation)
    os.environ["TABULAR_REPRESENTATION"] = tabular_representation
    os.environ["MUSHROOM_REPRESENTATION"] = tabular_representation
    os.environ["HTRU2_REPRESENTATION"] = tabular_representation
    os.environ["CHESS_REPRESENTATION"] = tabular_representation
    os.environ["CDC_DIABETES_REPRESENTATION"] = cdc_representation
    os.environ["CDC_DIABETES_SEMANTIC_FALLBACK"] = (
        "1" if getattr(args, "cdc_semantic_allow_transformed_fallback", True) else "0"
    )
    cfg.dataset_dir = args.dataset_dir
    if tabular_representation != "obfuscated":
        cfg.dataset_dir = os.path.join(args.dataset_dir, f"tabular_representation_{tabular_representation}")
    elif cdc_representation != "obfuscated":
        cfg.dataset_dir = os.path.join(args.dataset_dir, f"cdc_representation_{cdc_representation}")
    cfg.model = args.model
    cfg.provider = args.provider
    cfg.api_mode = args.api_mode
    if args.api_key is not None:
        cfg.api_key = args.api_key
        cfg.api_key_explicit = True
    if args.api_base_url is not None:
        cfg.api_base_url = args.api_base_url
    if args.azure_endpoint is not None:
        cfg.azure_endpoint = args.azure_endpoint
    if args.api_version is not None:
        cfg.api_version = args.api_version
    cfg.max_output_tokens = args.max_output_tokens
    cfg.reasoning_effort = args.reasoning_effort
    cfg.thinking_level = args.reasoning_effort
    cfg.verbosity = args.verbosity
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.tool_choice = args.tool_choice
    cfg.allow_tools = args.allow_tools
    cfg.enable_code_interpreter = args.enable_code_interpreter
    cfg.concurrency = args.concurrency
    cfg.per_call_timeout_s = args.timeout
    cfg.num_trials = args.num_trials
    return cfg


def build_boost_config(args: argparse.Namespace) -> BoostConfig:
    max_weak_error = args.max_weak_error
    min_alpha = args.min_alpha
    if args.strict_acceptance:
        max_weak_error = min(max_weak_error, 0.35)
        min_alpha = max(min_alpha, 0.05)
    tabular_representation = getattr(args, "tabular_representation", "obfuscated")
    cdc_representation = getattr(args, "cdc_representation", None) or default_cdc_representation(tabular_representation)

    return BoostConfig(
        boost_rounds=args.boost_rounds,
        batch_sizes=list(args.batch_sizes),
        round_retries=args.round_retries,
        max_weak_error=max_weak_error,
        min_alpha=min_alpha,
        num_trials=args.num_trials,
        stop_on_perfect_train=args.stop_on_perfect_train,
        resample_each_retry=args.resample_each_retry,
        output_dir=args.output_dir,
        repair_rounds=args.repair_rounds,
        repair_mistake_limit=args.repair_mistake_limit,
        repair_anchor_count=args.repair_anchor_count,
        repair_trigger_batch_acc_below=args.repair_trigger_batch_acc_below,
        repair_trigger_weighted_error_above=args.repair_trigger_weighted_error_above,
        repair_trigger_min_mistakes=args.repair_trigger_min_mistakes,
        sample_without_replacement=args.sample_without_replacement,
        strict_acceptance=args.strict_acceptance,
        whole_train_repair_rounds=args.whole_train_repair_rounds,
        whole_train_repair_batch_size=args.whole_train_repair_batch_size,
        whole_train_mistake_frac=args.whole_train_mistake_frac,
        whole_train_recent_fix_frac=args.whole_train_recent_fix_frac,
        whole_train_anchor_frac=args.whole_train_anchor_frac,
        tabular_representation=tabular_representation,
        cdc_representation=cdc_representation,
        accept_best_on_failure=args.accept_best_on_failure,
        best_fallback_max_weak_error=args.best_fallback_max_weak_error,
    )


async def main_async(args: argparse.Namespace) -> int:
    log = setup_logger(args.log_level)
    base_cfg = build_base_config(args)
    boost_cfg = build_boost_config(args)
    ensure_dir(boost_cfg.output_dir)

    client = base_runner.Runner(base_cfg, log)
    summary_rows: List[Dict[str, Any]] = []
    attempt_rows: List[Dict[str, Any]] = []

    try:
        for fn in base_cfg.functions:
            if fn not in base_runner.FUNCTION_NAME_MAPPING:
                raise ValueError(f"Unknown function id: {fn}")

            target_name = base_runner.FUNCTION_NAME_MAPPING[fn]
            if target_name != "cdc_diabetes":
                log.warning(
                    "non_cdc_target_requested",
                    extra={"fn": fn, "target_name": target_name},
                )

            for length in base_cfg.lengths:
                train_lines, val_lines, test_lines, is_decimal, is_tabular = client.ds.get(fn, length)
                for batch_size in boost_cfg.batch_sizes:
                    for trial_idx in range(1, boost_cfg.num_trials + 1):
                        log.info(
                            "boost_trial_start",
                            extra={
                                "fn": fn,
                                "target_name": target_name,
                                "length": length,
                                "batch_size": batch_size,
                                "trial": trial_idx,
                                "boost_rounds": boost_cfg.boost_rounds,
                            },
                        )

                        summary, trial_attempt_rows, accepted_rounds = await run_boosting_trial(
                            client,
                            log,
                            fn,
                            length,
                            target_name,
                            train_lines,
                            val_lines,
                            test_lines,
                            is_decimal,
                            is_tabular,
                            boost_cfg,
                            batch_size,
                            trial_idx,
                        )

                        summary_rows.append(summary)
                        attempt_rows.extend(trial_attempt_rows)

                        run_dir = os.path.join(
                            boost_cfg.output_dir,
                            target_name,
                            f"L{length}",
                            f"batch{batch_size}",
                            f"trial{trial_idx}",
                        )
                        ensure_dir(run_dir)

                        manifest = {
                            "summary": summary,
                            "accepted_rounds": accepted_rounds,
                            "config": {
                                "model": base_cfg.model,
                                "provider": client.provider,
                                "train_size": base_cfg.train_size,
                                "val_size": base_cfg.val_size,
                                "test_size": base_cfg.test_size,
                                "seed": base_cfg.seed,
                                "boost_rounds": boost_cfg.boost_rounds,
                                "batch_size": batch_size,
                                "round_retries": boost_cfg.round_retries,
                                "max_weak_error": boost_cfg.max_weak_error,
                                "min_alpha": boost_cfg.min_alpha,
                                "strict_acceptance": boost_cfg.strict_acceptance,
                                "tabular_representation": boost_cfg.tabular_representation,
                                "sample_without_replacement": boost_cfg.sample_without_replacement,
                                "repair_rounds": boost_cfg.repair_rounds,
                                "repair_mistake_limit": boost_cfg.repair_mistake_limit,
                                "repair_anchor_count": boost_cfg.repair_anchor_count,
                                "repair_trigger_batch_acc_below": boost_cfg.repair_trigger_batch_acc_below,
                                "repair_trigger_weighted_error_above": boost_cfg.repair_trigger_weighted_error_above,
                                "repair_trigger_min_mistakes": boost_cfg.repair_trigger_min_mistakes,
                                "whole_train_repair_rounds": boost_cfg.whole_train_repair_rounds,
                                "whole_train_repair_batch_size": boost_cfg.whole_train_repair_batch_size,
                                "whole_train_mistake_frac": boost_cfg.whole_train_mistake_frac,
                                "whole_train_recent_fix_frac": boost_cfg.whole_train_recent_fix_frac,
                                "whole_train_anchor_frac": boost_cfg.whole_train_anchor_frac,
                                "cdc_representation": boost_cfg.cdc_representation,
                                "accept_best_on_failure": boost_cfg.accept_best_on_failure,
                                "best_fallback_max_weak_error": boost_cfg.best_fallback_max_weak_error,
                                "dataset_dir": base_cfg.dataset_dir,
                            },
                        }
                        _safe_write_json(os.path.join(run_dir, "manifest.json"), manifest)
                        write_jsonl(os.path.join(run_dir, "attempts.jsonl"), trial_attempt_rows)

                        if accepted_rounds:
                            module_text = build_ensemble_module(accepted_rounds)
                            _safe_write_text(os.path.join(run_dir, "ensemble.py"), module_text)

                        log.info(
                            "boost_trial_done",
                            extra={
                                "fn": fn,
                                "target_name": target_name,
                                "length": length,
                                "batch_size": batch_size,
                                "trial": trial_idx,
                                "accepted_rounds": summary["accepted_rounds"],
                                "final_train_acc": summary["final_train_acc"],
                                "final_test_acc": summary["final_test_acc"],
                                "stopped_reason": summary["stopped_reason"],
                            },
                        )
    finally:
        await client.aclose()

    write_csv(os.path.join(boost_cfg.output_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(boost_cfg.output_dir, "attempts.csv"), attempt_rows)
    write_jsonl(os.path.join(boost_cfg.output_dir, "attempts.jsonl"), attempt_rows)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AdaBoost-style symbolic program synthesis experiments.",
    )
    p.add_argument("--functions", nargs="*", default=["fn_o"], help="Function ids. CDC diabetes is fn_o.")
    p.add_argument("--lengths", nargs="*", type=int, default=[21], help="Sequence lengths / tabular feature counts.")
    p.add_argument("--train-size", type=int, default=200, help="Train split size.")
    p.add_argument("--val-size", type=int, default=0, help="Validation split size. Defaults to 0 for train/test only.")
    p.add_argument("--test-size", type=int, default=1000, help="Test split size.")
    p.add_argument("--seed", type=int, default=42, help="Dataset seed.")
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset cache directory.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for summaries and saved ensembles.")
    p.add_argument(
        "--tabular-representation",
        choices=["obfuscated", "semantic", "hybrid"],
        default=os.getenv("TABULAR_REPRESENTATION", "obfuscated"),
        help="Input representation for non-CDC tabular datasets. semantic uses named fields; hybrid keeps names plus z-scores/code tokens.",
    )
    p.add_argument(
        "--cdc-representation",
        choices=["obfuscated", "semantic"],
        default=os.getenv("CDC_DIABETES_REPRESENTATION"),
        help="Optional CDC-specific representation override. Defaults to semantic when --tabular-representation is semantic or hybrid.",
    )
    p.add_argument(
        "--no-cdc-semantic-transformed-fallback",
        dest="cdc_semantic_allow_transformed_fallback",
        action="store_false",
        help="Fail semantic CDC generation if raw UCI/cache data is unavailable instead of using transformed bootstrap fallback.",
    )
    p.set_defaults(cdc_semantic_allow_transformed_fallback=True)

    p.add_argument("--boost-rounds", "--T", dest="boost_rounds", type=int, default=8, help="Maximum number of boosting rounds.")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128], help="Weighted train batch sizes to test.")
    p.add_argument("--round-retries", type=int, default=3, help="How many proposals to try for each round before giving up.")
    p.add_argument("--max-weak-error", type=float, default=0.499, help="Maximum weighted train error for accepting a learner.")
    p.add_argument("--min-alpha", type=float, default=1e-6, help="Reject learners with alpha below this value.")
    p.add_argument("--strict-acceptance", action="store_true", help="Clamp acceptance to max_weak_error<=0.35 and min_alpha>=0.05.")
    p.add_argument("--num-trials", type=int, default=3, help="Trials per batch size.")
    p.add_argument("--resample-each-retry", action="store_true", help="Resample a new weighted batch for each retry within a round.")
    p.add_argument("--sample-without-replacement", action="store_true", help="Sample weighted train batches without replacement.")
    p.add_argument("--stop-on-perfect-train", action="store_true", help="Stop a trial once ensemble train accuracy reaches 1.0.")
    p.add_argument("--repair-rounds", type=int, default=0, help="LLM repair iterations to run after each initial weak learner proposal.")
    p.add_argument("--repair-mistake-limit", type=int, default=128, help="Maximum misclassified batch examples to include in each repair prompt.")
    p.add_argument("--repair-anchor-count", type=int, default=16, help="Maximum correctly classified anchor examples to include in each repair prompt.")
    p.add_argument("--repair-trigger-batch-acc-below", type=float, default=None, help="Only run batch repair when initial batch_acc is below this threshold.")
    p.add_argument("--repair-trigger-weighted-error-above", type=float, default=None, help="Only run batch repair when initial weighted train error is above this threshold.")
    p.add_argument("--repair-trigger-min-mistakes", type=int, default=1, help="Minimum number of initial batch mistakes required before running repair.")
    p.add_argument("--whole-train-repair-rounds", type=int, default=0, help="Additional repair iterations that resample from whole-train mistakes/recent-fixes/anchors.")
    p.add_argument("--whole-train-repair-batch-size", type=int, default=256, help="Prompt example budget for each whole-train repair iteration.")
    p.add_argument("--whole-train-mistake-frac", type=float, default=0.7, help="Fraction of whole-train repair prompt reserved for current mistakes.")
    p.add_argument("--whole-train-recent-fix-frac", type=float, default=0.2, help="Fraction of whole-train repair prompt reserved for recently fixed examples.")
    p.add_argument("--whole-train-anchor-frac", type=float, default=0.1, help="Fraction of whole-train repair prompt reserved for correct anchors.")
    p.add_argument(
        "--accept-best-on-failure",
        action="store_true",
        help="If no retry satisfies max_weak_error/min_alpha, accept the best valid weak learner instead of failing the round.",
    )
    p.add_argument(
        "--best-fallback-max-weak-error",
        type=float,
        default=0.499,
        help="Maximum weighted error allowed for --accept-best-on-failure fallback candidates.",
    )

    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5"), help="Model name.")
    p.add_argument("--provider", default=os.getenv("PROVIDER", "auto"), help="Provider override.")
    p.add_argument("--api-mode", choices=["responses", "chat_completions"], default=os.getenv("API_MODE", "responses"))
    p.add_argument("--api-key", help="API key override for OpenAI-compatible or TAMU/Azure endpoints.")
    p.add_argument("--api-base-url", help="OpenAI-compatible base URL override.")
    p.add_argument("--azure-endpoint", help="Azure/TAMU endpoint override.")
    p.add_argument("--api-version", help="Azure/TAMU API version override.")
    p.add_argument("--max-output-tokens", type=int, default=int(os.getenv("MAX_OUTPUT_TOKENS", "20000")))
    p.add_argument("--reasoning-effort", default=os.getenv("REASONING_EFFORT", "high"))
    p.add_argument("--verbosity", default=os.getenv("TEXT_VERBOSITY", "low"))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE")) if os.getenv("TEMPERATURE") else None)
    p.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P")) if os.getenv("TOP_P") else None)
    p.add_argument("--tool-choice", default=os.getenv("TOOL_CHOICE", "auto"))
    p.add_argument("--allow-tools", dest="allow_tools", action="store_true")
    p.add_argument("--no-tools", dest="allow_tools", action="store_false")
    p.set_defaults(allow_tools=os.getenv("ALLOW_TOOLS", "1") == "1")
    p.add_argument("--enable-code-interpreter", action="store_true", help="Enable code interpreter where supported.")
    p.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "1")))
    p.add_argument("--timeout", type=float, default=float(os.getenv("PER_CALL_TIMEOUT_S", "1200")))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
