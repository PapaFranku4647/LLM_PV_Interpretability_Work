from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from program_synthesis.code_normalizer import sanitize_generated_code
    from program_synthesis.code1_verifier import build_code1_with_verification, compile_code1
    from program_synthesis.live_eval_common import (
        TARGET_NAME_BY_FN,
        compile_callable,
        detect_repo_root,
        extract_text_from_response,
        has_docstring,
        load_best_row,
        load_env,
        normalize_pred_to01,
        parse_json_from_text,
        parse_tabular_input,
    )
    from program_synthesis.prompt_variants import (
        build_thesis_generation_prompt,
        format_sample_for_thesis_prompt,
    )
except ModuleNotFoundError:
    # Allow running as: python run_step22_live_once.py from program_synthesis/
    from code_normalizer import sanitize_generated_code
    from code1_verifier import build_code1_with_verification, compile_code1  # type: ignore
    from live_eval_common import (  # type: ignore
        TARGET_NAME_BY_FN,
        compile_callable,
        detect_repo_root,
        extract_text_from_response,
        has_docstring,
        load_best_row,
        load_env,
        normalize_pred_to01,
        parse_json_from_text,
        parse_tabular_input,
    )
    from prompt_variants import (  # type: ignore
        build_thesis_generation_prompt,
        format_sample_for_thesis_prompt,
    )


def resolve_run_root(run_root_arg: str, repo_root: Path) -> Path:
    p = Path(run_root_arg)
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        repo_root / p,
        repo_root / "program_synthesis" / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_root / p


def get_first_test_sample(run_root: Path, row: dict[str, Any]) -> str:
    fn = row.get("fn")
    if fn not in TARGET_NAME_BY_FN:
        raise ValueError(f"Unsupported fn for tabular decode: {fn}")
    length = row.get("length")
    dataset_seed = row.get("dataset_seed")
    if length is None or dataset_seed is None:
        raise ValueError("Row missing length or dataset_seed.")

    target = TARGET_NAME_BY_FN[fn]
    test_path = run_root / "datasets" / target / f"L{length}" / f"seed{dataset_seed}" / "test.txt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    for raw in test_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and "->" in line:
            return line
    raise ValueError("No usable test sample found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live Step 2.2 thesis-generation request.")
    parser.add_argument(
        "--run-root",
        default="program_synthesis/runs_phase1_fn_o_prompt_compare_20260213",
        help="Run root containing *_trial*.jsonl and datasets/",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5"), help="Model for live request.")
    parser.add_argument("--max-output-tokens", type=int, default=2400, help="Responses API max_output_tokens.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "medium", "high"],
        default="minimal",
        help="Responses API reasoning effort.",
    )
    parser.add_argument(
        "--text-verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Responses API text verbosity.",
    )
    parser.add_argument(
        "--code1-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5"),
        help="Model for Code1 generation from raw thesis.",
    )
    parser.add_argument(
        "--code1-verifier-model",
        default="",
        help="Model for semantic Code1 verification (default: same as --code1-model).",
    )
    parser.add_argument(
        "--code1-max-output-tokens",
        type=int,
        default=1200,
        help="Responses API max_output_tokens for Code1 writer/verifier calls.",
    )
    parser.add_argument(
        "--code1-reasoning-effort",
        choices=["minimal", "medium", "high"],
        default="minimal",
        help="Responses API reasoning effort for Code1 writer/verifier calls.",
    )
    parser.add_argument(
        "--code1-text-verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Responses API text verbosity for Code1 writer/verifier calls.",
    )
    parser.add_argument(
        "--code1-exec-timeout",
        type=float,
        default=1.0,
        help="Per-testcase timeout (seconds) when executing Code1.",
    )
    parser.add_argument(
        "--code1-no-retry",
        action="store_true",
        help="Disable one retry attempt for Code1 generation/verification.",
    )
    args = parser.parse_args()

    repo_root = detect_repo_root()
    load_env(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing in env/.env")

    run_root = resolve_run_root(args.run_root, repo_root)
    best_file, row = load_best_row(run_root)

    original_code = row.get("code")
    if not isinstance(original_code, str) or not original_code.strip():
        raise SystemExit("Best row has no code.")
    sanitized_code = sanitize_generated_code(original_code)
    fn_callable = compile_callable(sanitized_code)

    sample_line = get_first_test_sample(run_root, row)
    x_str, y_str = [s.strip() for s in sample_line.split("->", 1)]
    sample = parse_tabular_input(x_str)
    true_label = int(y_str)
    pred_label = normalize_pred_to01(fn_callable(sample))

    sample_repr = format_sample_for_thesis_prompt(sample)
    prompt = build_thesis_generation_prompt(sanitized_code, sample_repr, pred_label)

    client = OpenAI(api_key=api_key)
    request_body = {
        "model": args.model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
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
    incomplete_details = getattr(response, "incomplete_details", None)
    if hasattr(incomplete_details, "model_dump"):
        incomplete_details = incomplete_details.model_dump()

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
    code1_verifier_model = args.code1_verifier_model.strip() or args.code1_model
    if has_conditions:
        try:
            code1_bundle = build_code1_with_verification(
                client=client,
                writer_model=args.code1_model,
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

    code1_bundle_dict = asdict(code1_bundle) if code1_bundle is not None else None
    code1_compile_ok = False
    code1_compile_error = None
    if code1_bundle is not None and isinstance(code1_bundle.final_code1, str) and code1_bundle.final_code1.strip():
        _, code1_compile_error = compile_code1(code1_bundle.final_code1)
        code1_compile_ok = code1_compile_error is None

    out_dir = repo_root / "program_synthesis" / f"quick_verify_step22_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "response.txt").write_text(response_text, encoding="utf-8")
    (out_dir / "sanitized_code.py").write_text(sanitized_code, encoding="utf-8")
    (out_dir / "raw_response.json").write_text(json.dumps(response_dump, indent=2), encoding="utf-8")
    code1_artifact = None
    if code1_bundle is not None and isinstance(code1_bundle.final_code1, str) and code1_bundle.final_code1.strip():
        code1_artifact = out_dir / "code1.py"
        code1_artifact.write_text(code1_bundle.final_code1, encoding="utf-8")

    summary = {
        "best_row_file": best_file,
        "row_meta": {
            "fn": row.get("fn"),
            "length": row.get("length"),
            "prompt_variant": row.get("prompt_variant"),
            "global_seed": row.get("global_seed"),
            "trial": row.get("trial"),
            "attempt": row.get("attempt"),
            "dataset_seed": row.get("dataset_seed"),
            "val_acc": row.get("val_acc"),
            "test_acc": row.get("test_acc"),
        },
        "sample_line": sample_line,
        "true_label": true_label,
        "predicted_label": pred_label,
        "checks": {
            "sanitized_code_has_hash_comment_char": "#" in sanitized_code,
            "sanitized_code_has_docstring": has_docstring(sanitized_code),
            "prompt_placeholders_removed": all(t not in prompt for t in ("[CODE0]", "[SAMPLE]", "[LABEL]")),
            "response_json_parse_error": parse_err,
            "response_has_conditions": has_conditions,
            "response_has_label": has_label,
            "response_label_matches_prediction": label_matches,
            "code1_compile_ok": code1_compile_ok,
            "code1_compile_error": code1_compile_error,
            "code1_semantic_judgement": (
                code1_bundle.semantic_result.judgement if code1_bundle and code1_bundle.semantic_result else None
            ),
            "code1_testcases_total": (
                code1_bundle.testcase_result.total if code1_bundle and code1_bundle.testcase_result else None
            ),
            "code1_testcases_failed": (
                code1_bundle.testcase_result.failed if code1_bundle and code1_bundle.testcase_result else None
            ),
            "code1_accepted": bool(code1_bundle and code1_bundle.accepted),
            "code1_attempts": code1_bundle.attempts if code1_bundle else None,
            "code1_verification_error": code1_error or (code1_bundle.error if code1_bundle else None),
        },
        "artifacts": {
            "dir": str(out_dir),
            "summary_json": str(out_dir / "summary.json"),
            "prompt_txt": str(out_dir / "prompt.txt"),
            "response_txt": str(out_dir / "response.txt"),
            "sanitized_code_py": str(out_dir / "sanitized_code.py"),
            "raw_response_json": str(out_dir / "raw_response.json"),
            "code1_py": str(code1_artifact) if code1_artifact else None,
        },
        "response_meta": {
            "id": response_id,
            "status": response_status,
            "incomplete_details": incomplete_details,
        },
        "request_meta": {
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "text_verbosity": args.text_verbosity,
            "code1_model": args.code1_model,
            "code1_verifier_model": code1_verifier_model,
            "code1_max_output_tokens": args.code1_max_output_tokens,
            "code1_reasoning_effort": args.code1_reasoning_effort,
            "code1_text_verbosity": args.code1_text_verbosity,
            "code1_exec_timeout": args.code1_exec_timeout,
            "code1_retry_enabled": not args.code1_no_retry,
        },
        "parsed_response_json": parsed_json,
        "code1_verification": code1_bundle_dict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\n--- RESPONSE_TEXT ---")
    print(response_text)


if __name__ == "__main__":
    main()
