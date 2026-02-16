from __future__ import annotations

import argparse
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

try:
    from program_synthesis.code_normalizer import sanitize_generated_code
    from program_synthesis.prompt_variants import (
        build_thesis_generation_prompt,
        format_sample_for_thesis_prompt,
    )
except ModuleNotFoundError:
    # Allow running as: python run_step22_live_once.py from program_synthesis/
    from code_normalizer import sanitize_generated_code
    from prompt_variants import (  # type: ignore
        build_thesis_generation_prompt,
        format_sample_for_thesis_prompt,
    )

TARGET_NAME_BY_FN = {
    "fn_m": "adult_income",
    "fn_n": "mushroom",
    "fn_o": "cdc_diabetes",
    "fn_p": "htru2",
    "fn_q": "chess",
}


def detect_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        script_dir,
        script_dir.parent,
    ]
    for c in candidates:
        if (c / "program_synthesis").exists() and (c / ".env").exists():
            return c
    if script_dir.name == "program_synthesis":
        return script_dir.parent
    return Path.cwd()


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


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_tabular_input(x_str: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in x_str.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def normalize_pred_to01(pred: Any) -> int:
    try:
        if hasattr(pred, "item"):
            pred = pred.item()
    except Exception:
        pass

    if isinstance(pred, bool):
        return 1 if pred else 0
    if isinstance(pred, int):
        return 1 if pred != 0 else 0
    if isinstance(pred, str):
        s = pred.strip().strip("\"'")
        if s in ("0", "1"):
            return int(s)
        sl = s.lower()
        if sl in ("true", "false"):
            return 1 if sl == "true" else 0
        try:
            v = int(float(s))
            return 1 if v != 0 else 0
        except Exception:
            return 1 if len(s) > 0 else 0
    return 1 if pred else 0


def compile_callable(code: str):
    tree = ast.parse(code)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found.")
    prefer = "f" if "f" in fn_names else fn_names[0]
    local_ns: Dict[str, Any] = {}
    exec(compile(tree, "<generated>", "exec"), {"__builtins__": __builtins__}, local_ns)
    fn = local_ns.get(prefer)
    if not callable(fn):
        raise ValueError(f"Function '{prefer}' not found after exec.")
    return fn


def has_docstring(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    def body_has_doc(body: list[ast.stmt]) -> bool:
        if not body:
            return False
        first = body[0]
        if not isinstance(first, ast.Expr):
            return False
        v = first.value
        return (isinstance(v, ast.Constant) and isinstance(v.value, str)) or isinstance(v, ast.Str)

    if body_has_doc(tree.body):
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if body_has_doc(node.body):
                return True
    return False


def parse_json_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty_response_text"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_is_not_object"
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed, None
            return None, "fenced_json_is_not_object"
        except Exception:
            pass

    any_obj = re.search(r"\{[\s\S]*\}", text)
    if any_obj:
        try:
            parsed = json.loads(any_obj.group(0))
            if isinstance(parsed, dict):
                return parsed, None
            return None, "extracted_json_is_not_object"
        except Exception as e:
            return None, f"json_parse_error: {e}"
    return None, "no_json_object_found"


def _obj_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_text_from_response(response: Any) -> str:
    direct = _obj_get(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = _obj_get(response, "output") or []
    for item in output:
        content = _obj_get(item, "content") or []
        if isinstance(content, list):
            for c in content:
                c_type = _obj_get(c, "type")
                if c_type in {"output_text", "text"}:
                    text_val = _obj_get(c, "text")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val.strip())
                    elif isinstance(text_val, dict):
                        nested = text_val.get("value")
                        if isinstance(nested, str) and nested.strip():
                            parts.append(nested.strip())
    return "\n".join(parts).strip()


def load_best_row(run_root: Path) -> Tuple[str, Dict[str, Any]]:
    # Select Code0 by validation accuracy only to avoid test-set selection leakage.
    if not run_root.exists():
        raise ValueError(f"Run root not found: {run_root}")
    best_val_acc: Optional[float] = None
    best_row = None
    best_file = None

    for path in sorted(run_root.glob("*_trial*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                val_acc = row.get("val_acc")
                if val_acc is None:
                    continue
                try:
                    val_acc_f = float(val_acc)
                except Exception:
                    continue
                if best_val_acc is None or val_acc_f > best_val_acc:
                    best_val_acc = val_acc_f
                    best_row = row
                    best_file = path.name

    if best_row is None or best_file is None:
        raise ValueError("No compile-valid row found with val_acc.")
    return best_file, best_row


def get_first_test_sample(run_root: Path, row: Dict[str, Any]) -> str:
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

    out_dir = repo_root / "program_synthesis" / f"quick_verify_step22_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "response.txt").write_text(response_text, encoding="utf-8")
    (out_dir / "sanitized_code.py").write_text(sanitized_code, encoding="utf-8")
    (out_dir / "raw_response.json").write_text(json.dumps(response_dump, indent=2), encoding="utf-8")

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
        },
        "artifacts": {
            "dir": str(out_dir),
            "summary_json": str(out_dir / "summary.json"),
            "prompt_txt": str(out_dir / "prompt.txt"),
            "response_txt": str(out_dir / "response.txt"),
            "sanitized_code_py": str(out_dir / "sanitized_code.py"),
            "raw_response_json": str(out_dir / "raw_response.json"),
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
        },
        "parsed_response_json": parsed_json,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\n--- RESPONSE_TEXT ---")
    print(response_text)


if __name__ == "__main__":
    main()
