from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_SYNTHESIS_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

import runner as base_runner  # noqa: E402
import boosted_runner  # noqa: E402


@dataclass
class CandidateProgram:
    row_index: int
    attempt: int
    round_idx: int
    retry: int
    code: str
    code_sha256: str
    fn_callable: Any
    train_preds_pm: List[int]
    val_preds_pm: List[int]
    test_preds_pm: List[int]
    train_eval_errors: int
    val_eval_errors: int
    test_eval_errors: int
    original_row: Dict[str, Any]


@dataclass
class SelectedCandidate:
    step: int
    candidate: CandidateProgram
    alpha: float
    direction: int
    weighted_error: float
    train_acc: float
    val_acc: float
    test_acc: float


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


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        _safe_write_text(path, "")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
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
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if body:
        body += "\n"
    _safe_write_text(path, body)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("posthoc_selector")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.handlers[:] = [handler]
    logger.propagate = False
    return logger


def configure_tabular_environment(tabular_representation: str, cdc_representation: str) -> None:
    os.environ["TABULAR_REPRESENTATION"] = tabular_representation
    os.environ["MUSHROOM_REPRESENTATION"] = tabular_representation
    os.environ["HTRU2_REPRESENTATION"] = tabular_representation
    os.environ["CHESS_REPRESENTATION"] = tabular_representation
    os.environ["CDC_DIABETES_REPRESENTATION"] = cdc_representation
    os.environ.setdefault("CDC_DIABETES_SEMANTIC_FALLBACK", "1")


def resolve_dataset_dir(dataset_dir: str, tabular_representation: str, cdc_representation: str) -> str:
    if tabular_representation != "obfuscated":
        return os.path.join(dataset_dir, f"tabular_representation_{tabular_representation}")
    if cdc_representation != "obfuscated":
        return os.path.join(dataset_dir, f"cdc_representation_{cdc_representation}")
    return dataset_dir


def load_dataset_examples(
    *,
    fn: str,
    length: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    dataset_dir: str,
    tabular_representation: str,
    cdc_representation: Optional[str],
    log: logging.Logger,
) -> Tuple[List[boosted_runner.Example], List[boosted_runner.Example], List[boosted_runner.Example], bool]:
    resolved_cdc_representation = cdc_representation or boosted_runner.default_cdc_representation(tabular_representation)
    configure_tabular_environment(tabular_representation, resolved_cdc_representation)

    cfg = base_runner.Config()
    cfg.train_size = train_size
    cfg.val_size = val_size
    cfg.test_size = test_size
    cfg.seed = seed
    cfg.dataset_dir = resolve_dataset_dir(dataset_dir, tabular_representation, resolved_cdc_representation)

    store = base_runner.DatasetStore(cfg, log)
    train_lines, val_lines, test_lines, _is_decimal, is_tabular = store.get(fn, length)
    train_examples = boosted_runner.parse_examples(train_lines, is_tabular)
    val_examples = boosted_runner.parse_examples(val_lines, is_tabular)
    test_examples = boosted_runner.parse_examples(test_lines, is_tabular)
    return train_examples, val_examples, test_examples, is_tabular


def load_attempt_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def predict_pm_for_examples(fn_callable: Any, examples: Sequence[boosted_runner.Example]) -> Tuple[List[int], int]:
    preds: List[int] = []
    eval_errors = 0
    for example in examples:
        try:
            pred01 = boosted_runner.predict_01(fn_callable, example.x)
        except Exception:
            pred01 = 1 - example.y01
            eval_errors += 1
        preds.append(1 if pred01 == 1 else -1)
    return preds, eval_errors


def accuracy_from_preds(preds_pm: Sequence[int], examples: Sequence[boosted_runner.Example]) -> float:
    if not examples:
        return 0.0
    correct = 0
    for pred_pm, example in zip(preds_pm, examples):
        pred01 = 1 if pred_pm == 1 else 0
        correct += int(pred01 == example.y01)
    return correct / len(examples)


def accuracy_from_scores(scores: Sequence[float], examples: Sequence[boosted_runner.Example]) -> float:
    if not examples:
        return 0.0
    correct = 0
    for score, example in zip(scores, examples):
        pred01 = 1 if score >= 0.0 else 0
        correct += int(pred01 == example.y01)
    return correct / len(examples)


def weighted_error_from_preds(
    preds_pm: Sequence[int],
    labels_pm: Sequence[int],
    weights: Sequence[float],
    direction: int,
) -> float:
    total_weight = float(sum(weights)) or 1.0
    weighted_errors = 0.0
    for pred_pm, label_pm, weight in zip(preds_pm, labels_pm, weights):
        if direction * pred_pm != label_pm:
            weighted_errors += weight
    return weighted_errors / total_weight


def update_weights_from_preds(
    weights: Sequence[float],
    labels_pm: Sequence[int],
    preds_pm: Sequence[int],
    alpha: float,
    direction: int,
) -> List[float]:
    directed_preds = [direction * pred_pm for pred_pm in preds_pm]
    return boosted_runner.update_distribution(weights, labels_pm, directed_preds, alpha)


def load_candidate_programs(
    rows: Sequence[Dict[str, Any]],
    train_examples: Sequence[boosted_runner.Example],
    val_examples: Sequence[boosted_runner.Example],
    test_examples: Sequence[boosted_runner.Example],
    *,
    dedupe: bool = True,
) -> Tuple[List[CandidateProgram], List[Dict[str, Any]]]:
    candidates: List[CandidateProgram] = []
    rejected: List[Dict[str, Any]] = []
    seen_hashes = set()

    for row_index, row in enumerate(rows):
        code = row.get("candidate_code")
        if not code:
            rejected.append(
                {
                    "row_index": row_index,
                    "attempt": row.get("attempt"),
                    "reason": "missing_candidate_code",
                }
            )
            continue

        code_sha256 = str(row.get("candidate_code_sha256") or boosted_runner._hash_text(code))
        if dedupe and code_sha256 in seen_hashes:
            rejected.append(
                {
                    "row_index": row_index,
                    "attempt": row.get("attempt"),
                    "candidate_code_sha256": code_sha256,
                    "reason": "duplicate_code",
                }
            )
            continue

        try:
            fn_callable = base_runner.compile_callable_from_code(code)
            train_preds, train_errors = predict_pm_for_examples(fn_callable, train_examples)
            val_preds, val_errors = predict_pm_for_examples(fn_callable, val_examples)
            test_preds, test_errors = predict_pm_for_examples(fn_callable, test_examples)
        except Exception as exc:
            rejected.append(
                {
                    "row_index": row_index,
                    "attempt": row.get("attempt"),
                    "candidate_code_sha256": code_sha256,
                    "reason": "compile_or_eval_error",
                    "error": str(exc),
                }
            )
            continue

        seen_hashes.add(code_sha256)
        candidates.append(
            CandidateProgram(
                row_index=row_index,
                attempt=int(row.get("attempt") or row_index + 1),
                round_idx=int(row.get("round") or 0),
                retry=int(row.get("retry") or 0),
                code=code,
                code_sha256=code_sha256,
                fn_callable=fn_callable,
                train_preds_pm=train_preds,
                val_preds_pm=val_preds,
                test_preds_pm=test_preds,
                train_eval_errors=train_errors,
                val_eval_errors=val_errors,
                test_eval_errors=test_errors,
                original_row=dict(row),
            )
        )
    return candidates, rejected


def candidate_score_rows(
    candidates: Sequence[CandidateProgram],
    train_examples: Sequence[boosted_runner.Example],
    val_examples: Sequence[boosted_runner.Example],
    test_examples: Sequence[boosted_runner.Example],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "row_index": candidate.row_index,
                "attempt": candidate.attempt,
                "round": candidate.round_idx,
                "retry": candidate.retry,
                "candidate_code_sha256": candidate.code_sha256,
                "candidate_code_chars": len(candidate.code),
                "individual_train_acc": accuracy_from_preds(candidate.train_preds_pm, train_examples),
                "individual_val_acc": accuracy_from_preds(candidate.val_preds_pm, val_examples),
                "individual_test_acc": accuracy_from_preds(candidate.test_preds_pm, test_examples),
                "train_eval_errors": candidate.train_eval_errors,
                "val_eval_errors": candidate.val_eval_errors,
                "test_eval_errors": candidate.test_eval_errors,
                "original_weighted_error": candidate.original_row.get("weighted_error"),
                "original_alpha": candidate.original_row.get("alpha"),
                "original_batch_acc": candidate.original_row.get("batch_acc"),
                "original_candidate_selected": candidate.original_row.get("candidate_selected"),
            }
        )
    return rows


def greedy_select_ensemble(
    candidates: Sequence[CandidateProgram],
    train_examples: Sequence[boosted_runner.Example],
    val_examples: Sequence[boosted_runner.Example],
    test_examples: Sequence[boosted_runner.Example],
    *,
    max_rounds: int,
    max_weak_error: float,
    min_alpha: float,
    min_val_improvement: float,
    allow_inverted_candidates: bool,
) -> Tuple[List[SelectedCandidate], List[Dict[str, Any]]]:
    if not candidates:
        return [], []

    labels_pm = [example.ypm for example in train_examples]
    weights = [1.0 / len(train_examples)] * len(train_examples)
    train_scores = [0.0] * len(train_examples)
    val_scores = [0.0] * len(val_examples)
    test_scores = [0.0] * len(test_examples)
    selected: List[SelectedCandidate] = []
    trace: List[Dict[str, Any]] = []
    used_keys = set()
    best_val_acc = 0.0
    directions = [1, -1] if allow_inverted_candidates else [1]

    for step in range(1, max_rounds + 1):
        best: Optional[SelectedCandidate] = None
        best_candidate_scores: Optional[Tuple[List[float], List[float], List[float]]] = None
        considered = 0
        valid = 0

        for candidate in candidates:
            if candidate.code_sha256 in used_keys:
                continue
            for direction in directions:
                considered += 1
                weighted_error = weighted_error_from_preds(
                    candidate.train_preds_pm,
                    labels_pm,
                    weights,
                    direction,
                )
                if weighted_error <= 0.0:
                    weighted_error = 1e-12
                if weighted_error >= max_weak_error:
                    continue
                alpha = 0.5 * math.log((1.0 - weighted_error) / weighted_error)
                if not math.isfinite(alpha) or alpha < min_alpha:
                    continue
                valid += 1

                candidate_train_scores = [
                    current + alpha * direction * pred_pm
                    for current, pred_pm in zip(train_scores, candidate.train_preds_pm)
                ]
                candidate_val_scores = [
                    current + alpha * direction * pred_pm
                    for current, pred_pm in zip(val_scores, candidate.val_preds_pm)
                ]
                candidate_test_scores = [
                    current + alpha * direction * pred_pm
                    for current, pred_pm in zip(test_scores, candidate.test_preds_pm)
                ]
                train_acc = accuracy_from_scores(candidate_train_scores, train_examples)
                val_acc = accuracy_from_scores(candidate_val_scores, val_examples)
                test_acc = accuracy_from_scores(candidate_test_scores, test_examples)

                proposal = SelectedCandidate(
                    step=step,
                    candidate=candidate,
                    alpha=alpha,
                    direction=direction,
                    weighted_error=weighted_error,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    test_acc=test_acc,
                )
                if best is None or (
                    proposal.val_acc,
                    proposal.train_acc,
                    -proposal.weighted_error,
                    proposal.test_acc,
                ) > (
                    best.val_acc,
                    best.train_acc,
                    -best.weighted_error,
                    best.test_acc,
                ):
                    best = proposal
                    best_candidate_scores = (
                        candidate_train_scores,
                        candidate_val_scores,
                        candidate_test_scores,
                    )

        trace_row: Dict[str, Any] = {
            "step": step,
            "considered": considered,
            "valid": valid,
            "current_best_val_acc": best_val_acc,
        }

        if best is None or best_candidate_scores is None:
            trace_row["stopped_reason"] = "no_valid_candidate"
            trace.append(trace_row)
            break

        trace_row.update(
            {
                "best_row_index": best.candidate.row_index,
                "best_attempt": best.candidate.attempt,
                "best_round": best.candidate.round_idx,
                "best_retry": best.candidate.retry,
                "best_direction": best.direction,
                "best_alpha": best.alpha,
                "best_weighted_error": best.weighted_error,
                "best_train_acc": best.train_acc,
                "best_val_acc": best.val_acc,
                "best_test_acc": best.test_acc,
                "best_candidate_code_sha256": best.candidate.code_sha256,
            }
        )

        if best.val_acc <= best_val_acc + min_val_improvement:
            trace_row["stopped_reason"] = "no_validation_improvement"
            trace.append(trace_row)
            break

        selected.append(best)
        used_keys.add(best.candidate.code_sha256)
        train_scores, val_scores, test_scores = best_candidate_scores
        weights = update_weights_from_preds(
            weights,
            labels_pm,
            best.candidate.train_preds_pm,
            best.alpha,
            best.direction,
        )
        best_val_acc = best.val_acc
        trace_row["selected"] = True
        trace.append(trace_row)

    return selected, trace


def uniform_greedy_select_ensemble(
    candidates: Sequence[CandidateProgram],
    train_examples: Sequence[boosted_runner.Example],
    val_examples: Sequence[boosted_runner.Example],
    test_examples: Sequence[boosted_runner.Example],
    *,
    max_rounds: int,
    min_val_improvement: float,
    allow_inverted_candidates: bool,
) -> Tuple[List[SelectedCandidate], List[Dict[str, Any]]]:
    if not candidates:
        return [], []

    labels_pm = [example.ypm for example in train_examples]
    train_scores = [0.0] * len(train_examples)
    val_scores = [0.0] * len(val_examples)
    test_scores = [0.0] * len(test_examples)
    selected: List[SelectedCandidate] = []
    trace: List[Dict[str, Any]] = []
    used_keys = set()
    best_val_acc = 0.0
    directions = [1, -1] if allow_inverted_candidates else [1]

    for step in range(1, max_rounds + 1):
        best: Optional[SelectedCandidate] = None
        best_candidate_scores: Optional[Tuple[List[float], List[float], List[float]]] = None
        considered = 0

        for candidate in candidates:
            if candidate.code_sha256 in used_keys:
                continue
            for direction in directions:
                considered += 1
                candidate_train_scores = [
                    current + direction * pred_pm
                    for current, pred_pm in zip(train_scores, candidate.train_preds_pm)
                ]
                candidate_val_scores = [
                    current + direction * pred_pm
                    for current, pred_pm in zip(val_scores, candidate.val_preds_pm)
                ]
                candidate_test_scores = [
                    current + direction * pred_pm
                    for current, pred_pm in zip(test_scores, candidate.test_preds_pm)
                ]
                train_acc = accuracy_from_scores(candidate_train_scores, train_examples)
                val_acc = accuracy_from_scores(candidate_val_scores, val_examples)
                test_acc = accuracy_from_scores(candidate_test_scores, test_examples)
                weighted_error = weighted_error_from_preds(
                    candidate.train_preds_pm,
                    labels_pm,
                    [1.0 / len(train_examples)] * len(train_examples),
                    direction,
                )

                proposal = SelectedCandidate(
                    step=step,
                    candidate=candidate,
                    alpha=1.0,
                    direction=direction,
                    weighted_error=weighted_error,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    test_acc=test_acc,
                )
                if best is None or (
                    proposal.val_acc,
                    proposal.train_acc,
                    proposal.test_acc,
                    -proposal.weighted_error,
                ) > (
                    best.val_acc,
                    best.train_acc,
                    best.test_acc,
                    -best.weighted_error,
                ):
                    best = proposal
                    best_candidate_scores = (
                        candidate_train_scores,
                        candidate_val_scores,
                        candidate_test_scores,
                    )

        trace_row: Dict[str, Any] = {
            "step": step,
            "considered": considered,
            "valid": considered,
            "current_best_val_acc": best_val_acc,
        }

        if best is None or best_candidate_scores is None:
            trace_row["stopped_reason"] = "no_candidate"
            trace.append(trace_row)
            break

        trace_row.update(
            {
                "best_row_index": best.candidate.row_index,
                "best_attempt": best.candidate.attempt,
                "best_round": best.candidate.round_idx,
                "best_retry": best.candidate.retry,
                "best_direction": best.direction,
                "best_alpha": best.alpha,
                "best_weighted_error": best.weighted_error,
                "best_train_acc": best.train_acc,
                "best_val_acc": best.val_acc,
                "best_test_acc": best.test_acc,
                "best_candidate_code_sha256": best.candidate.code_sha256,
            }
        )

        if best.val_acc <= best_val_acc + min_val_improvement:
            trace_row["stopped_reason"] = "no_validation_improvement"
            trace.append(trace_row)
            break

        selected.append(best)
        used_keys.add(best.candidate.code_sha256)
        train_scores, val_scores, test_scores = best_candidate_scores
        best_val_acc = best.val_acc
        trace_row["selected"] = True
        trace.append(trace_row)

    return selected, trace


def selected_rows(selected: Sequence[SelectedCandidate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in selected:
        rows.append(
            {
                "step": item.step,
                "row_index": item.candidate.row_index,
                "attempt": item.candidate.attempt,
                "round": item.candidate.round_idx,
                "retry": item.candidate.retry,
                "direction": item.direction,
                "alpha": item.alpha,
                "weighted_error": item.weighted_error,
                "train_acc": item.train_acc,
                "val_acc": item.val_acc,
                "test_acc": item.test_acc,
                "candidate_code_sha256": item.candidate.code_sha256,
                "candidate_code_chars": len(item.candidate.code),
            }
        )
    return rows


def build_posthoc_ensemble_module(selected: Sequence[SelectedCandidate]) -> str:
    parts: List[str] = [
        '"""Auto-generated post-hoc boosted ensemble wrapper."""',
        "from typing import Any",
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
    ]

    for idx, item in enumerate(selected, start=1):
        code = textwrap.dedent(item.candidate.code).strip()
        fn_name = boosted_runner.extract_function_name(code)
        parts.append(f"# Learner {idx}: row {item.candidate.row_index}, attempt {item.candidate.attempt}")
        parts.append(code)
        parts.append(f"h_{idx} = {fn_name}")
        parts.append("")

    alpha_str = ", ".join(f"{item.alpha:.16g}" for item in selected)
    direction_str = ", ".join(str(item.direction) for item in selected)
    learner_refs = ", ".join(f"h_{idx}" for idx in range(1, len(selected) + 1))
    parts.extend(
        [
            f"ALPHAS = [{alpha_str}]",
            f"DIRECTIONS = [{direction_str}]",
            f"LEARNERS = [{learner_refs}]",
            "",
            "def f(x: Any) -> int:",
            "    score = 0.0",
            "    for alpha, direction, learner in zip(ALPHAS, DIRECTIONS, LEARNERS):",
            "        score += alpha * direction * _normalize_pred_to_pm1(learner(x))",
            "    return 1 if score >= 0.0 else 0",
            "",
        ]
    )
    return "\n".join(parts)


def run_posthoc_selection(args: argparse.Namespace, log: logging.Logger) -> Dict[str, Any]:
    train_examples, val_examples, test_examples, is_tabular = load_dataset_examples(
        fn=args.function,
        length=args.length,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        tabular_representation=args.tabular_representation,
        cdc_representation=args.cdc_representation,
        log=log,
    )
    if not is_tabular:
        raise ValueError("posthoc_selector currently expects tabular parsed examples.")

    attempt_rows = load_attempt_rows(args.attempts_jsonl)
    candidates, rejected = load_candidate_programs(
        attempt_rows,
        train_examples,
        val_examples,
        test_examples,
        dedupe=not args.no_dedupe,
    )
    if args.selection_mode == "weighted_greedy":
        selected, trace = greedy_select_ensemble(
            candidates,
            train_examples,
            val_examples,
            test_examples,
            max_rounds=args.max_rounds,
            max_weak_error=args.max_weak_error,
            min_alpha=args.min_alpha,
            min_val_improvement=args.min_val_improvement,
            allow_inverted_candidates=args.allow_inverted_candidates,
        )
    else:
        selected, trace = uniform_greedy_select_ensemble(
            candidates,
            train_examples,
            val_examples,
            test_examples,
            max_rounds=args.max_rounds,
            min_val_improvement=args.min_val_improvement,
            allow_inverted_candidates=args.allow_inverted_candidates,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    scores = candidate_score_rows(candidates, train_examples, val_examples, test_examples)
    selected_summary_rows = selected_rows(selected)
    final = selected[-1] if selected else None
    summary = {
        "attempts_jsonl": args.attempts_jsonl,
        "function": args.function,
        "length": args.length,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "tabular_representation": args.tabular_representation,
        "cdc_representation": args.cdc_representation or boosted_runner.default_cdc_representation(args.tabular_representation),
        "loaded_attempt_rows": len(attempt_rows),
        "compiled_candidates": len(candidates),
        "rejected_candidates": len(rejected),
        "selected_count": len(selected),
        "selection_mode": args.selection_mode,
        "max_rounds": args.max_rounds,
        "max_weak_error": args.max_weak_error,
        "min_alpha": args.min_alpha,
        "min_val_improvement": args.min_val_improvement,
        "allow_inverted_candidates": args.allow_inverted_candidates,
        "final_train_acc": final.train_acc if final else None,
        "final_val_acc": final.val_acc if final else None,
        "final_test_acc": final.test_acc if final else None,
        "stopped_reason": trace[-1].get("stopped_reason") if trace else "no_candidates",
    }

    write_csv(os.path.join(args.output_dir, "candidate_scores.csv"), scores)
    write_csv(os.path.join(args.output_dir, "selection_trace.csv"), trace)
    write_csv(os.path.join(args.output_dir, "selected.csv"), selected_summary_rows)
    write_csv(os.path.join(args.output_dir, "summary.csv"), [summary])
    write_jsonl(
        os.path.join(args.output_dir, "selected_candidates.jsonl"),
        [
            {
                **row,
                "candidate_code": item.candidate.code,
                "original_row": item.candidate.original_row,
            }
            for row, item in zip(selected_summary_rows, selected)
        ],
    )
    write_jsonl(os.path.join(args.output_dir, "rejected_candidates.jsonl"), rejected)
    _safe_write_json(
        os.path.join(args.output_dir, "manifest.json"),
        {
            "summary": summary,
            "selected": selected_summary_rows,
            "trace": trace,
            "config": vars(args),
        },
    )
    if selected:
        _safe_write_text(os.path.join(args.output_dir, "ensemble.py"), build_posthoc_ensemble_module(selected))

    log.info(
        "posthoc_selection_done",
        extra={
            "compiled_candidates": len(candidates),
            "selected_count": len(selected),
            "final_val_acc": summary["final_val_acc"],
            "final_test_acc": summary["final_test_acc"],
            "output_dir": args.output_dir,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a post-hoc validation-selected ensemble from logged CodeBoost candidates.")
    parser.add_argument("--attempts-jsonl", required=True, help="Path to attempts.jsonl containing candidate_code fields.")
    parser.add_argument("--output-dir", required=True, help="Directory where post-hoc selection artifacts will be written.")
    parser.add_argument("--function", default="fn_o", help="Function id used to regenerate/load the dataset split.")
    parser.add_argument("--length", type=int, default=21, help="Sequence length / tabular feature count.")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", default=boosted_runner.DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--tabular-representation",
        choices=["obfuscated", "semantic", "hybrid"],
        default="semantic",
    )
    parser.add_argument("--cdc-representation", choices=["obfuscated", "semantic"], default=None)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=["weighted_greedy", "uniform_greedy"],
        default="weighted_greedy",
        help="weighted_greedy recomputes AdaBoost alphas; uniform_greedy greedily adds +/- one-vote candidates by validation.",
    )
    parser.add_argument("--max-weak-error", type=float, default=0.499)
    parser.add_argument("--min-alpha", type=float, default=1e-6)
    parser.add_argument("--min-val-improvement", type=float, default=0.0)
    parser.add_argument("--allow-inverted-candidates", action="store_true", help="Also consider each candidate with its predictions inverted.")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not remove duplicate candidate code hashes before selection.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logger(args.log_level)
    run_posthoc_selection(args, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
