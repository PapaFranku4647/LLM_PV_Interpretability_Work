from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Optional


TARGET_NAME_BY_FN = {
    "fn_m": "adult_income",
    "fn_n": "mushroom",
    "fn_o": "cdc_diabetes",
    "fn_p": "htru2",
    "fn_q": "chess",
}

FEATURE_LENGTH_BY_FN = {
    "fn_m": 14,
    "fn_n": 20,
    "fn_o": 21,
    "fn_p": 8,
    "fn_q": 35,
}


def compute_auto_split(
    n_positive: int,
    n_negative: int,
    train_cap: int = 200,
    train_ratio: float = 0.10,
    val_ratio: float = 0.15,
    total_cap: int = 5000,
) -> tuple[int, int, int]:
    pool = min(n_positive, n_negative)
    total = min(2 * pool, total_cap)

    raw_train = int(total * train_ratio)
    train = min(raw_train, train_cap)
    overflow = raw_train - train

    val = int(total * val_ratio) + overflow
    test = total - train - val

    if train % 2:
        train -= 1
        test += 1
    if val % 2:
        val -= 1
        test += 1
    if test % 2:
        test -= 1

    return train, val, test


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


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_tabular_input(x_str: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def parse_tabular_line(line: str) -> tuple[dict[str, Any], int]:
    x_str, y_str = [s.strip() for s in line.split("->", 1)]
    return parse_tabular_input(x_str), int(y_str)


def sample_dict_to_ordered_list(sample: dict[str, Any]) -> list[Any]:
    ordered_pairs: list[tuple[int, Any]] = []
    for key, value in sample.items():
        if not isinstance(key, str) or not key.startswith("x"):
            continue
        try:
            idx = int(key[1:])
        except Exception:
            continue
        ordered_pairs.append((idx, value))
    if not ordered_pairs:
        return []
    max_idx = max(i for i, _ in ordered_pairs)
    out: list[Any] = [None] * (max_idx + 1)
    for idx, value in ordered_pairs:
        out[idx] = value
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


def _sample_to_kv_string(sample: dict[str, Any]) -> str:
    """Reconstruct 'x0:val0, x1:val1, ...' string from sample dict."""
    ordered_pairs: list[tuple[int, str, Any]] = []
    for key, value in sample.items():
        if not isinstance(key, str) or not key.startswith("x"):
            continue
        try:
            idx = int(key[1:])
        except Exception:
            continue
        ordered_pairs.append((idx, key, value))
    ordered_pairs.sort()
    return ", ".join(f"{k}:{v}" for _, k, v in ordered_pairs)


def predict_code0_label(code0_fn: Any, sample: dict[str, Any]) -> tuple[int, str]:
    attempts: list[tuple[str, Any]] = [("dict", sample)]
    ordered = sample_dict_to_ordered_list(sample)
    if ordered:
        attempts.append(("list", ordered))
        attempts.append(("tuple", tuple(ordered)))
    kv_str = _sample_to_kv_string(sample)
    if kv_str:
        attempts.append(("string", kv_str))

    errors: list[str] = []
    for mode, payload in attempts:
        try:
            pred = code0_fn(payload)
            return normalize_pred_to01(pred), mode
        except Exception as e:
            errors.append(f"{mode}:{e}")
    raise RuntimeError("code0_eval_failed: " + " | ".join(errors))


def compile_callable(code: str) -> Any:
    tree = ast.parse(code)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found.")
    prefer = "f" if "f" in fn_names else fn_names[0]
    local_ns: dict[str, Any] = {}
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


def parse_json_from_text(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty_response_text"
    # Strip <think>...</think> blocks (e.g., from Qwen, DeepSeek reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return None, "empty_after_think_strip"
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


def load_best_row(run_root: Path) -> tuple[str, dict[str, Any]]:
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
