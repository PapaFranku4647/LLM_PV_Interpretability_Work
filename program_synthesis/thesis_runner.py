from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import subprocess
import sys
import tokenize
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from openai import OpenAI

try:
    from program_synthesis.code_normalizer import sanitize_generated_code
    from program_synthesis.code1_verifier import build_code1_with_verification, compile_code1
    from program_synthesis.live_eval_common import (
        FEATURE_LENGTH_BY_FN,
        TARGET_NAME_BY_FN,
        compile_callable,
        compute_auto_split,
        detect_repo_root,
        extract_text_from_response,
        has_docstring,
        load_best_row,
        load_env,
        parse_json_from_text,
        parse_tabular_line,
        predict_code0_label,
    )
    from program_synthesis.prompt_variants import (
        build_thesis_generation_prompt,
        build_thesis_generation_prompt_v2,
        format_sample_for_thesis_prompt,
    )
    from program_synthesis.thesis_evaluator import ThesisEvaluator, load_split_lines
except ModuleNotFoundError:
    from live_eval_common import (  # type: ignore
        FEATURE_LENGTH_BY_FN,
        TARGET_NAME_BY_FN,
        compile_callable,
        compute_auto_split,
        detect_repo_root,
        extract_text_from_response,
        has_docstring,
        load_best_row,
        load_env,
        parse_json_from_text,
        parse_tabular_line,
        predict_code0_label,
    )
    from code_normalizer import sanitize_generated_code  # type: ignore
    from code1_verifier import build_code1_with_verification, compile_code1  # type: ignore
    from prompt_variants import build_thesis_generation_prompt, build_thesis_generation_prompt_v2, format_sample_for_thesis_prompt  # type: ignore
    from thesis_evaluator import ThesisEvaluator, load_split_lines  # type: ignore


def has_comment_tokens(code: str) -> bool:
    if not code:
        return False
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                return True
    except Exception:
        # Fail-safe: if tokenization fails, treat as suspicious.
        return True
    return False


def read_split_lines(path: Path) -> list[str]:
    return load_split_lines(path)


def compute_equation_metrics(
    code0_fn: Any,
    code1_fn: Any,
    sample: dict[str, Any],
    pred_label: int,
    train_lines: Sequence[str],
) -> dict[str, Any]:
    evaluator = ThesisEvaluator(
        code0_fn=code0_fn,
        train_lines=train_lines,
        parse_line_fn=parse_tabular_line,
        predict_code0_label_fn=predict_code0_label,
    )
    result = evaluator.evaluate_thesis(
        sample_x=sample,
        pred_label=pred_label,
        check_conditions_fn=code1_fn,
    )
    return result.to_legacy_dict()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step23_matrix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def run_runner_val_selection(
    python_exe: str,
    repo_root: Path,
    fn: str,
    seed: int,
    run_dir: Path,
    dataset_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "results.jsonl"
    out_csv = run_dir / "results.csv"
    out_manifest = run_dir / "results_manifest.json"
    cmd = [
        python_exe,
        str(repo_root / "program_synthesis" / "runner_val_selection.py"),
        "--functions",
        fn,
        "--attempts",
        str(args.attempts),
        "--num-trials",
        str(args.num_trials),
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(seed),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--tool-choice",
        "none",
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--prompt-variant",
        args.prompt_variant,
        "--dataset-dir",
        str(dataset_dir),
        "--out-jsonl",
        str(out_jsonl),
        "--out-csv",
        str(out_csv),
        "--out-manifest",
        str(out_manifest),
    ]
    logger.info("runner_start fn=%s seed=%s run_dir=%s", fn, seed, run_dir)
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (run_dir / "runner_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (run_dir / "runner_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner_val_selection failed fn={fn} seed={seed} returncode={proc.returncode}. "
            f"See {run_dir / 'runner_stderr.log'}"
        )
    logger.info("runner_done fn=%s seed=%s", fn, seed)


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accepted_rate = (
        (sum(1 for r in rows if bool(r.get("code1_accepted"))) / n)
        if n > 0
        else 0.0
    )
    compile_ok_rate = (
        (sum(1 for r in rows if bool(r.get("code1_compile_ok"))) / n)
        if n > 0
        else 0.0
    )
    label_match_rate = (
        (sum(1 for r in rows if bool(r.get("response_label_matches_prediction"))) / n)
        if n > 0
        else 0.0
    )

    metric_results = [ThesisEvaluator.result_from_mapping(r) for r in rows]
    metric_report = ThesisEvaluator.summarize(metric_results)
    out = metric_report.to_legacy_dict()
    out.update(
        {
            "accepted_rate": accepted_rate,
            "compile_ok_rate": compile_ok_rate,
            "label_match_rate": label_match_rate,
        }
    )
    return out


def _short_code_preview(code: Optional[str], max_len: int = 240) -> Optional[str]:
    if not isinstance(code, str):
        return None
    compact = code.strip().replace("\r\n", "\n")
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "...<truncated>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.3 live matrix runner with equation metrics.")
    parser.add_argument("--functions", nargs="+", default=["fn_m", "fn_n", "fn_o", "fn_p", "fn_q"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2201, 2202, 2203])
    parser.add_argument("--samples-per-seed", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--prompt-variant", default="explain", choices=["standard", "explain", "interview", "preview"])

    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--reasoning-effort", default="minimal", choices=["minimal", "medium", "high"])
    parser.add_argument("--text-verbosity", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--max-output-tokens", type=int, default=1400)

    parser.add_argument("--code1-model", default="")
    parser.add_argument("--code1-verifier-model", default="")
    parser.add_argument("--code1-reasoning-effort", default="minimal", choices=["minimal", "medium", "high"])
    parser.add_argument("--code1-text-verbosity", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--code1-max-output-tokens", type=int, default=1200)
    parser.add_argument("--code1-exec-timeout", type=float, default=1.0)
    parser.add_argument("--code1-no-retry", action="store_true")

    parser.add_argument(
        "--out-root",
        default="program_synthesis/runs_step23_live_matrix",
        help="Directory where matrix artifacts are written.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Dataset cache directory passed to runner_val_selection (default: <out-root>/datasets).",
    )
    parser.add_argument("--thesis-prompt-version", default="v1", choices=["v1", "v2"],
        help="Thesis prompt version: v1 (original) or v2 (code-tracing + coverage guidance).")
    parser.add_argument("--compute-baselines", action="store_true",
        help="Compute trivial baselines per (fn, seed) and include in overall_summary.json.")
    parser.add_argument("--skip-runner", action="store_true", help="Reuse existing run dirs; skip runner_val_selection calls.")
    parser.add_argument("--auto-split", action="store_true",
        help="Auto-compute train/val/test sizes per dataset using balanced class pools.")
    parser.add_argument("--train-cap", type=int, default=200,
        help="Max train size when --auto-split is on (default: 200).")
    parser.add_argument("--total-cap", type=int, default=5000,
        help="Max total samples (train+val+test) when --auto-split is on (default: 5000).")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used for runner_val_selection subprocess calls.",
    )
    args = parser.parse_args()

    repo_root = detect_repo_root()
    load_env(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing in env/.env")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = repo_root / args.out_root / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_root / "matrix.log")
    logger.info("matrix_start out_root=%s", out_root)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (out_root / "datasets")
    if not dataset_dir.is_absolute():
        dataset_dir = (repo_root / dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    code1_model = args.code1_model.strip() or args.model
    code1_verifier_model = args.code1_verifier_model.strip() or code1_model
    client = OpenAI(api_key=api_key)

    case_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []

    auto_split_sizes: dict[str, dict[str, int]] = {}

    for fn in args.functions:
        if fn not in TARGET_NAME_BY_FN:
            raise SystemExit(f"Unsupported fn: {fn}. Supported: {sorted(TARGET_NAME_BY_FN)}")
        if args.auto_split:
            try:
                from src.data_handler import get_available_class_counts
            except ModuleNotFoundError:
                sys.path.insert(0, str(repo_root / "src"))
                from data_handler import get_available_class_counts  # type: ignore
            target = TARGET_NAME_BY_FN[fn]
            L = FEATURE_LENGTH_BY_FN[fn]
            n_pos, n_neg = get_available_class_counts(target, L)
            args.train_size, args.val_size, args.test_size = compute_auto_split(
                n_pos, n_neg, train_cap=args.train_cap, total_cap=args.total_cap)
            auto_split_sizes[fn] = {
                "n_positive": n_pos, "n_negative": n_neg,
                "pool": min(n_pos, n_neg),
                "train": args.train_size, "val": args.val_size, "test": args.test_size,
            }
            logger.info("auto_split fn=%s pool=%d train=%d val=%d test=%d",
                fn, min(n_pos, n_neg), args.train_size, args.val_size, args.test_size)
        for seed in args.seeds:
            combo_id = f"{fn}_seed{seed}"
            run_dir = out_root / "runs" / combo_id
            case_root = out_root / "cases" / combo_id
            case_root.mkdir(parents=True, exist_ok=True)
            logger.info("combo_start fn=%s seed=%s", fn, seed)

            if not args.skip_runner:
                run_runner_val_selection(
                    python_exe=args.python_exe,
                    repo_root=repo_root,
                    fn=fn,
                    seed=seed,
                    run_dir=run_dir,
                    dataset_dir=dataset_dir,
                    args=args,
                    logger=logger,
                )

            best_row_file, row = load_best_row(run_dir)
            original_code = row.get("code")
            if not isinstance(original_code, str) or not original_code.strip():
                raise RuntimeError(f"Best row has no code for fn={fn}, seed={seed}")
            sanitized_code = sanitize_generated_code(original_code)
            code0_has_hash = "#" in sanitized_code
            code0_has_comment_tokens = has_comment_tokens(sanitized_code)
            code0_has_docstring = has_docstring(sanitized_code)
            code0_comment_free_ok = (not code0_has_comment_tokens) and (not code0_has_docstring)
            if not code0_comment_free_ok:
                logger.error(
                    "code0_not_comment_free fn=%s seed=%s has_comment_tokens=%s has_docstring=%s",
                    fn,
                    seed,
                    code0_has_comment_tokens,
                    code0_has_docstring,
                )
                raise RuntimeError(f"Sanitized Code0 still contains comments/docstrings for fn={fn}, seed={seed}")
            code0_fn = compile_callable(sanitized_code)

            length = row.get("length")
            dataset_seed = row.get("dataset_seed")
            if length is None or dataset_seed is None:
                raise RuntimeError(f"Missing length/dataset_seed for fn={fn}, seed={seed}")

            target = TARGET_NAME_BY_FN[fn]
            split_dir = dataset_dir / target / f"L{length}" / f"seed{dataset_seed}"
            train_lines = read_split_lines(split_dir / "train.txt")
            test_lines = read_split_lines(split_dir / "test.txt")
            if args.samples_per_seed > len(test_lines):
                raise RuntimeError(
                    f"Requested samples_per_seed={args.samples_per_seed} but only {len(test_lines)} test rows "
                    f"for fn={fn}, seed={seed}"
                )

            for sample_idx in range(args.samples_per_seed):
                test_line = test_lines[sample_idx]
                sample, true_label = parse_tabular_line(test_line)
                pred_label = None
                code0_pred_mode = None
                code0_pred_error = None
                try:
                    pred_label, code0_pred_mode = predict_code0_label(code0_fn, sample)
                except Exception as e:
                    code0_pred_error = str(e)
                if pred_label is None:
                    case_dir = case_root / f"sample_{sample_idx + 1:04d}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    (case_dir / "code0_sanitized.py").write_text(sanitized_code, encoding="utf-8")
                    write_json(
                        case_dir / "summary.json",
                        {
                            "fn": fn,
                            "seed": seed,
                            "sample_index": sample_idx + 1,
                            "test_line": test_line,
                            "true_label": true_label,
                            "predicted_label": None,
                            "code0_pred_mode": None,
                            "code0_pred_error": code0_pred_error,
                            "code1_accepted": False,
                            "code1_compile_ok": False,
                            "code1_verification_error": "code0_prediction_failed",
                            "S_size": len(train_lines),
                            "A_S_size": 0,
                            "x_in_A": False,
                            "coverage_ratio": 0.0,
                            "coverage_eq": 0.0,
                            "agreement_count": 0,
                            "faithfulness": None,
                            "code0_comment_free_ok": code0_comment_free_ok,
                            "code0_has_hash_comment_char": code0_has_hash,
                            "code0_has_comment_tokens": code0_has_comment_tokens,
                            "code0_has_docstring": code0_has_docstring,
                            "case_dir": str(case_dir),
                        },
                    )
                    case_rows.append(
                        {
                            "fn": fn,
                            "seed": seed,
                            "sample_index": sample_idx + 1,
                            "run_dir": str(run_dir),
                            "best_row_file": best_row_file,
                            "row_trial": row.get("trial"),
                            "row_attempt": row.get("attempt"),
                            "dataset_seed": dataset_seed,
                            "length": length,
                            "prompt_variant": row.get("prompt_variant"),
                            "val_acc": row.get("val_acc"),
                            "test_acc": row.get("test_acc"),
                            "test_line": test_line,
                            "true_label": true_label,
                            "predicted_label": None,
                            "code0_pred_mode": None,
                            "code0_pred_error": code0_pred_error,
                            "code0_comment_free_ok": code0_comment_free_ok,
                            "code0_has_hash_comment_char": code0_has_hash,
                            "code0_has_comment_tokens": code0_has_comment_tokens,
                            "code0_has_docstring": code0_has_docstring,
                            "response_id": None,
                            "response_status": None,
                            "response_json_parse_error": "code0_prediction_failed",
                            "response_has_conditions": False,
                            "response_has_label": False,
                            "response_label_matches_prediction": False,
                            "thesis_conditions": None,
                            "thesis_label": None,
                            "code1_compile_ok": False,
                            "code1_compile_error": None,
                            "code1_accepted": False,
                            "code1_attempts": 0,
                            "code1_semantic_judgement": None,
                            "code1_testcases_total": None,
                            "code1_testcases_failed": None,
                            "code1_verification_error": "code0_prediction_failed",
                            "S_size": len(train_lines),
                            "A_S_size": 0,
                            "x_in_A": False,
                            "coverage_ratio": 0.0,
                            "coverage_eq": 0.0,
                            "agreement_count": 0,
                            "faithfulness": None,
                            "code0_eval_errors": 0,
                            "code1_eval_errors": 0,
                            "case_dir": str(case_dir),
                            "code1_path": None,
                            "code1_preview": None,
                            "code1_verification": None,
                        }
                    )
                    logger.warning(
                        "case_skipped_code0_eval fn=%s seed=%s sample=%s err=%s",
                        fn,
                        seed,
                        sample_idx + 1,
                        code0_pred_error,
                    )
                    continue
                sample_repr = format_sample_for_thesis_prompt(sample)
                if args.thesis_prompt_version == "v2":
                    thesis_prompt = build_thesis_generation_prompt_v2(sanitized_code, sample_repr, pred_label)
                else:
                    thesis_prompt = build_thesis_generation_prompt(sanitized_code, sample_repr, pred_label)

                request_body = {
                    "model": args.model,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": thesis_prompt}]}],
                    "reasoning": {"effort": args.reasoning_effort},
                    "text": {"verbosity": args.text_verbosity},
                    "max_output_tokens": args.max_output_tokens,
                    "tool_choice": "none",
                }
                response = client.responses.create(**request_body)
                response_text = extract_text_from_response(response)
                response_dump = response.model_dump() if hasattr(response, "model_dump") else {}
                response_status = getattr(response, "status", None)
                response_id = getattr(response, "id", None)

                parsed_json, parse_err = parse_json_from_text(response_text)
                has_conditions = (
                    isinstance(parsed_json, dict)
                    and isinstance(parsed_json.get("conditions"), str)
                    and bool(parsed_json["conditions"].strip())
                )
                has_label = isinstance(parsed_json, dict) and ("label" in parsed_json)
                label_matches = False
                if has_label:
                    try:
                        label_matches = int(parsed_json["label"]) == int(pred_label)
                    except Exception:
                        label_matches = False

                thesis_label_for_code1 = pred_label
                if has_label:
                    try:
                        thesis_label_for_code1 = int(parsed_json["label"])
                    except Exception:
                        thesis_label_for_code1 = pred_label

                code1_bundle = None
                code1_error = None
                code1_compile_ok = False
                code1_compile_error = None
                equation_metrics = {
                    "S_size": len(train_lines),
                    "A_S_size": 0,
                    "x_in_A": False,
                    "coverage_ratio": 0.0,
                    "coverage_eq": 0.0,
                    "agreement_count": 0,
                    "faithfulness": None,
                    "code0_eval_errors": 0,
                    "code1_eval_errors": 0,
                }
                if has_conditions:
                    try:
                        code1_bundle = build_code1_with_verification(
                            client=client,
                            writer_model=code1_model,
                            verifier_model=code1_verifier_model,
                            thesis_conditions=str(parsed_json["conditions"]).strip(),
                            thesis_label=thesis_label_for_code1,
                            sample_repr=sample_repr,
                            retry_once=not args.code1_no_retry,
                            max_output_tokens=args.code1_max_output_tokens,
                            reasoning_effort=args.code1_reasoning_effort,
                            text_verbosity=args.code1_text_verbosity,
                            execution_timeout_s=args.code1_exec_timeout,
                        )
                    except Exception as e:
                        code1_error = str(e)
                else:
                    code1_error = "missing_conditions"

                if code1_bundle is not None and isinstance(code1_bundle.final_code1, str) and code1_bundle.final_code1.strip():
                    code1_callable, code1_compile_error = compile_code1(code1_bundle.final_code1)
                    code1_compile_ok = code1_compile_error is None and callable(code1_callable)
                    if code1_compile_ok and callable(code1_callable):
                        equation_metrics = compute_equation_metrics(
                            code0_fn=code0_fn,
                            code1_fn=code1_callable,
                            sample=sample,
                            pred_label=pred_label,
                            train_lines=train_lines,
                        )

                case_dir = case_root / f"sample_{sample_idx + 1:04d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "code0_sanitized.py").write_text(sanitized_code, encoding="utf-8")
                (case_dir / "prompt.txt").write_text(thesis_prompt, encoding="utf-8")
                (case_dir / "response.txt").write_text(response_text, encoding="utf-8")
                write_json(case_dir / "raw_response.json", response_dump)
                code1_path = None
                code1_text = None
                if code1_bundle is not None and isinstance(code1_bundle.final_code1, str) and code1_bundle.final_code1.strip():
                    code1_path = case_dir / "code1.py"
                    code1_text = code1_bundle.final_code1
                    code1_path.write_text(code1_text, encoding="utf-8")

                row_out = {
                    "fn": fn,
                    "seed": seed,
                    "sample_index": sample_idx + 1,
                    "run_dir": str(run_dir),
                    "best_row_file": best_row_file,
                    "row_trial": row.get("trial"),
                    "row_attempt": row.get("attempt"),
                    "dataset_seed": dataset_seed,
                    "length": length,
                    "prompt_variant": row.get("prompt_variant"),
                    "val_acc": row.get("val_acc"),
                    "test_acc": row.get("test_acc"),
                    "test_line": test_line,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "code0_pred_mode": code0_pred_mode,
                    "code0_pred_error": code0_pred_error,
                    "code0_comment_free_ok": code0_comment_free_ok,
                    "code0_has_hash_comment_char": code0_has_hash,
                    "code0_has_comment_tokens": code0_has_comment_tokens,
                    "code0_has_docstring": code0_has_docstring,
                    "response_id": response_id,
                    "response_status": response_status,
                    "response_json_parse_error": parse_err,
                    "response_has_conditions": has_conditions,
                    "response_has_label": has_label,
                    "response_label_matches_prediction": label_matches,
                    "thesis_conditions": (parsed_json or {}).get("conditions") if isinstance(parsed_json, dict) else None,
                    "thesis_label": (parsed_json or {}).get("label") if isinstance(parsed_json, dict) else None,
                    "code1_compile_ok": code1_compile_ok,
                    "code1_compile_error": code1_compile_error,
                    "code1_accepted": bool(code1_bundle and code1_bundle.accepted),
                    "code1_attempts": (code1_bundle.attempts if code1_bundle else None),
                    "code1_semantic_judgement": (
                        code1_bundle.semantic_result.judgement if code1_bundle and code1_bundle.semantic_result else None
                    ),
                    "code1_testcases_total": (
                        code1_bundle.testcase_result.total if code1_bundle and code1_bundle.testcase_result else None
                    ),
                    "code1_testcases_failed": (
                        code1_bundle.testcase_result.failed if code1_bundle and code1_bundle.testcase_result else None
                    ),
                    "code1_verification_error": code1_error or (code1_bundle.error if code1_bundle else None),
                    "S_size": equation_metrics["S_size"],
                    "A_S_size": equation_metrics["A_S_size"],
                    "x_in_A": equation_metrics["x_in_A"],
                    "coverage_ratio": equation_metrics["coverage_ratio"],
                    "coverage_eq": equation_metrics["coverage_eq"],
                    "agreement_count": equation_metrics["agreement_count"],
                    "faithfulness": equation_metrics["faithfulness"],
                    "code0_eval_errors": equation_metrics["code0_eval_errors"],
                    "code1_eval_errors": equation_metrics["code1_eval_errors"],
                    "case_dir": str(case_dir),
                    "code1_path": str(code1_path) if code1_path else None,
                    "code1_preview": _short_code_preview(code1_text),
                    "code1_verification": asdict(code1_bundle) if code1_bundle else None,
                }
                case_rows.append(row_out)
                write_json(case_dir / "summary.json", row_out)

                logger.info(
                    "case_done fn=%s seed=%s sample=%s accepted=%s cov_eq=%.4f faith=%s code1_err=%s",
                    fn,
                    seed,
                    sample_idx + 1,
                    row_out["code1_accepted"],
                    float(row_out["coverage_eq"] or 0.0),
                    (
                        f"{float(row_out['faithfulness']):.4f}"
                        if row_out.get("faithfulness") is not None
                        else "None"
                    ),
                    row_out.get("code1_verification_error"),
                )

            combo_case_rows = [r for r in case_rows if r["fn"] == fn and int(r["seed"]) == int(seed)]
            combo_summary = {
                "fn": fn,
                "seed": seed,
                "run_dir": str(run_dir),
                "dataset_dir": str(dataset_dir),
                "samples_per_seed": args.samples_per_seed,
            }
            combo_summary.update(summarize_group(combo_case_rows))
            combo_rows.append(combo_summary)
            logger.info(
                "combo_done fn=%s seed=%s cases=%s accepted_rate=%.4f mean_cov_eq=%.4f mean_faith_defined=%s",
                fn,
                seed,
                combo_summary["n_cases"],
                combo_summary["accepted_rate"],
                combo_summary["mean_coverage_eq_all"],
                (
                    f"{combo_summary['mean_faithfulness_defined_only']:.4f}"
                    if combo_summary["mean_faithfulness_defined_only"] is not None
                    else "None"
                ),
            )

    per_fn_rows: list[dict[str, Any]] = []
    for fn in args.functions:
        rows_for_fn = [r for r in case_rows if r["fn"] == fn]
        fn_summary = {"fn": fn}
        fn_summary.update(summarize_group(rows_for_fn))
        per_fn_rows.append(fn_summary)

    baselines_data: Optional[dict[str, Any]] = None
    if args.compute_baselines:
        try:
            from thesis_analysis import compute_trivial_baselines as _compute_baselines
        except ModuleNotFoundError:
            from program_synthesis.thesis_analysis import compute_trivial_baselines as _compute_baselines  # type: ignore
        baselines_data = {}
        for fn in args.functions:
            for seed in args.seeds:
                combo_id = f"{fn}_seed{seed}"
                run_dir = out_root / "runs" / combo_id
                try:
                    best_row_file, row = load_best_row(run_dir)
                    original_code = row.get("code")
                    if not isinstance(original_code, str):
                        continue
                    code0_fn_bl = compile_callable(sanitize_generated_code(original_code))
                    target = TARGET_NAME_BY_FN[fn]
                    length = row.get("length")
                    dataset_seed = row.get("dataset_seed")
                    split_dir = dataset_dir / target / f"L{length}" / f"seed{dataset_seed}"
                    train_lines_bl = read_split_lines(split_dir / "train.txt")
                    baselines_data[combo_id] = _compute_baselines(train_lines_bl, code0_fn_bl)
                    logger.info("baselines_done combo=%s", combo_id)
                except Exception as e:
                    logger.warning("baselines_failed combo=%s err=%s", combo_id, e)

    overall_summary = summarize_group(case_rows)
    overall_payload = {
        "timestamp": stamp,
        "args": vars(args),
        "n_functions": len(args.functions),
        "n_seeds": len(args.seeds),
        "n_combos": len(args.functions) * len(args.seeds),
        "samples_per_seed": args.samples_per_seed,
        "expected_total_cases": len(args.functions) * len(args.seeds) * args.samples_per_seed,
        "actual_total_cases": len(case_rows),
        "overall": overall_summary,
        "per_function": per_fn_rows,
        "auto_split_sizes": auto_split_sizes if auto_split_sizes else None,
        "trivial_baselines": baselines_data,
    }

    write_jsonl(out_root / "cases.jsonl", case_rows)
    write_jsonl(out_root / "combo_summaries.jsonl", combo_rows)
    write_json(out_root / "overall_summary.json", overall_payload)

    per_fn_csv = out_root / "per_function_summary.csv"
    with per_fn_csv.open("w", newline="", encoding="utf-8") as f:
        if per_fn_rows:
            fieldnames = list(per_fn_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_fn_rows:
                writer.writerow(row)

    logger.info("matrix_done out_root=%s total_cases=%s", out_root, len(case_rows))
    print(json.dumps({"out_root": str(out_root), "overall": overall_summary}, indent=2))


if __name__ == "__main__":
    main()

