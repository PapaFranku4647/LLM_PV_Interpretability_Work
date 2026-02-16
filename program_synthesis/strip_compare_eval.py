from __future__ import annotations

import argparse
import ast
import io
import json
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


TARGET_NAME_BY_FN = {
    "fn_m": "adult_income",
    "fn_n": "mushroom",
    "fn_o": "cdc_diabetes",
    "fn_p": "htru2",
    "fn_q": "chess",
}

TABULAR_FNS = set(TARGET_NAME_BY_FN.keys())


def normalize_generated_code(code_str: str) -> str:
    normalized = textwrap.dedent((code_str or "").strip())
    if normalized.startswith("```"):
        normalized = re.sub(
            r"^```(?:python)?\s*|\s*```$",
            "",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return normalized


def strip_comment_tokens(code_str: str) -> str:
    try:
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(code_str).readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok)
        return tokenize.untokenize(out)
    except Exception:
        return code_str


class DocstringStripper(ast.NodeTransformer):
    @staticmethod
    def _strip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
        if not body:
            return body
        first = body[0]
        if not isinstance(first, ast.Expr):
            return body
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
        if isinstance(value, ast.Str):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node


def sanitize_generated_code(code_str: str) -> str:
    normalized = normalize_generated_code(code_str)
    no_comments = strip_comment_tokens(normalized)
    try:
        tree = ast.parse(no_comments)
    except Exception:
        return no_comments
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return no_comments


def compile_callable(code_str: str, strip_mode: bool) -> Callable[[Any], int]:
    code = sanitize_generated_code(code_str) if strip_mode else normalize_generated_code(code_str)
    tree = ast.parse(code)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found in generated code.")
    prefer_name = "f" if "f" in fn_names else fn_names[0]
    local_ns: Dict[str, Any] = {}
    exec(compile(tree, filename="<generated>", mode="exec"), {"__builtins__": __builtins__}, local_ns)
    fn = local_ns.get(prefer_name)
    if not callable(fn):
        raise ValueError(f"Function '{prefer_name}' not found after exec.")
    return fn


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


def parse_tabular_input(x_str: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in x_str.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            result[k] = float(v)
        except ValueError:
            result[k] = v
    return result


def evaluate_accuracy(
    fn_callable: Callable[[Any], int],
    data_lines: List[str],
    is_tabular: bool,
) -> Tuple[float, int]:
    if not data_lines:
        return 0.0, 0
    correct = 0
    errors = 0
    for line in data_lines:
        try:
            x, y = line.split("->")
            x = x.strip()
            y_int = int(y.strip())
            if is_tabular:
                x = parse_tabular_input(x)
            pred = fn_callable(x)
            pred_int = normalize_pred_to01(pred)
            correct += int(pred_int == y_int)
        except Exception:
            errors += 1
    return correct / len(data_lines), errors


def load_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            yield row


def to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


@dataclass
class CompareResult:
    runner_val: Optional[float]
    runner_test: Optional[float]
    orig_val: Optional[float]
    orig_test: Optional[float]
    strip_val: Optional[float]
    strip_test: Optional[float]
    orig_compile_error: Optional[str]
    strip_compile_error: Optional[str]
    orig_val_errors: int
    orig_test_errors: int
    strip_val_errors: int
    strip_test_errors: int


def compare_row(row: Dict[str, Any], run_root: Path) -> Optional[CompareResult]:
    code = row.get("code")
    if not isinstance(code, str) or not code.strip():
        return None

    fn = row.get("fn")
    length = row.get("length")
    dataset_seed = row.get("dataset_seed")
    if fn not in TARGET_NAME_BY_FN or dataset_seed is None or length is None:
        return None

    target_name = TARGET_NAME_BY_FN[fn]
    ds_dir = run_root / "datasets" / target_name / f"L{length}" / f"seed{dataset_seed}"
    val_path = ds_dir / "val.txt"
    test_path = ds_dir / "test.txt"
    if not val_path.exists() or not test_path.exists():
        return None

    val_lines = val_path.read_text(encoding="utf-8").splitlines()
    test_lines = test_path.read_text(encoding="utf-8").splitlines()
    is_tabular = fn in TABULAR_FNS

    runner_val = to_float_or_none(row.get("val_acc"))
    runner_test = to_float_or_none(row.get("test_acc"))

    orig_fn = None
    strip_fn = None
    orig_compile_error = None
    strip_compile_error = None

    try:
        orig_fn = compile_callable(code, strip_mode=False)
    except Exception as e:
        orig_compile_error = str(e)

    try:
        strip_fn = compile_callable(code, strip_mode=True)
    except Exception as e:
        strip_compile_error = str(e)

    orig_val = orig_test = None
    strip_val = strip_test = None
    orig_val_errors = orig_test_errors = 0
    strip_val_errors = strip_test_errors = 0

    if orig_fn is not None:
        orig_val, orig_val_errors = evaluate_accuracy(orig_fn, val_lines, is_tabular)
        orig_test, orig_test_errors = evaluate_accuracy(orig_fn, test_lines, is_tabular)

    if strip_fn is not None:
        strip_val, strip_val_errors = evaluate_accuracy(strip_fn, val_lines, is_tabular)
        strip_test, strip_test_errors = evaluate_accuracy(strip_fn, test_lines, is_tabular)

    return CompareResult(
        runner_val=runner_val,
        runner_test=runner_test,
        orig_val=orig_val,
        orig_test=orig_test,
        strip_val=strip_val,
        strip_test=strip_test,
        orig_compile_error=orig_compile_error,
        strip_compile_error=strip_compile_error,
        orig_val_errors=orig_val_errors,
        orig_test_errors=orig_test_errors,
        strip_val_errors=strip_val_errors,
        strip_test_errors=strip_test_errors,
    )


def same_float(a: Optional[float], b: Optional[float], eps: float = 1e-12) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= eps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runner metrics with original-vs-stripped code evaluation.")
    parser.add_argument("--run-root", required=True, help="Run directory, e.g. program_synthesis/runs_phase1_fn_o_prompt_compare_strip_20260216")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit after filtering (0 = no limit)")
    parser.add_argument("--variant", default="", help="Optional prompt variant filter, e.g. standard")
    parser.add_argument("--seed-min", type=int, default=None)
    parser.add_argument("--seed-max", type=int, default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    trial_files = sorted(run_root.glob("*_trial*.jsonl"))

    rows_scanned = 0
    rows_compared = 0
    runner_metric_rows = 0
    orig_compile_ok = 0
    strip_compile_ok = 0
    strip_matches_runner = 0
    orig_matches_runner = 0
    orig_vs_strip_diff = 0
    sum_runner_test = 0.0
    sum_orig_test = 0.0
    sum_strip_test = 0.0
    sum_runner_val = 0.0
    sum_orig_val = 0.0
    sum_strip_val = 0.0
    count_runner = 0
    count_orig = 0
    count_strip = 0
    diff_examples: List[Dict[str, Any]] = []

    for trial_path in trial_files:
        for row in load_jsonl_rows(trial_path):
            rows_scanned += 1
            if args.variant and (row.get("prompt_variant") != args.variant):
                continue
            seed = row.get("global_seed")
            if args.seed_min is not None and (seed is None or int(seed) < args.seed_min):
                continue
            if args.seed_max is not None and (seed is None or int(seed) > args.seed_max):
                continue

            compared = compare_row(row, run_root)
            if compared is None:
                continue
            rows_compared += 1

            if compared.runner_val is not None and compared.runner_test is not None:
                runner_metric_rows += 1
                sum_runner_val += compared.runner_val
                sum_runner_test += compared.runner_test
                count_runner += 1

            if compared.orig_compile_error is None and compared.orig_val is not None and compared.orig_test is not None:
                orig_compile_ok += 1
                sum_orig_val += compared.orig_val
                sum_orig_test += compared.orig_test
                count_orig += 1

            if compared.strip_compile_error is None and compared.strip_val is not None and compared.strip_test is not None:
                strip_compile_ok += 1
                sum_strip_val += compared.strip_val
                sum_strip_test += compared.strip_test
                count_strip += 1

            if same_float(compared.runner_val, compared.strip_val) and same_float(compared.runner_test, compared.strip_test):
                strip_matches_runner += 1

            if same_float(compared.runner_val, compared.orig_val) and same_float(compared.runner_test, compared.orig_test):
                orig_matches_runner += 1

            orig_strip_same = same_float(compared.orig_val, compared.strip_val) and same_float(compared.orig_test, compared.strip_test)
            if not orig_strip_same:
                orig_vs_strip_diff += 1
                if len(diff_examples) < 10:
                    diff_examples.append(
                        {
                            "file": trial_path.name,
                            "fn": row.get("fn"),
                            "variant": row.get("prompt_variant"),
                            "seed": row.get("global_seed"),
                            "trial": row.get("trial"),
                            "attempt": row.get("attempt"),
                            "runner_val": compared.runner_val,
                            "runner_test": compared.runner_test,
                            "orig_val": compared.orig_val,
                            "orig_test": compared.orig_test,
                            "strip_val": compared.strip_val,
                            "strip_test": compared.strip_test,
                            "orig_compile_error": compared.orig_compile_error,
                            "strip_compile_error": compared.strip_compile_error,
                            "orig_val_errors": compared.orig_val_errors,
                            "orig_test_errors": compared.orig_test_errors,
                            "strip_val_errors": compared.strip_val_errors,
                            "strip_test_errors": compared.strip_test_errors,
                        }
                    )

            if args.limit and rows_compared >= args.limit:
                break
        if args.limit and rows_compared >= args.limit:
            break

    summary = {
        "run_root": str(run_root),
        "trial_files": len(trial_files),
        "rows_scanned": rows_scanned,
        "rows_compared": rows_compared,
        "runner_metric_rows": runner_metric_rows,
        "orig_compile_ok": orig_compile_ok,
        "strip_compile_ok": strip_compile_ok,
        "strip_matches_runner": strip_matches_runner,
        "orig_matches_runner": orig_matches_runner,
        "orig_vs_strip_diff_rows": orig_vs_strip_diff,
        "mean_runner_val": (sum_runner_val / count_runner) if count_runner else None,
        "mean_runner_test": (sum_runner_test / count_runner) if count_runner else None,
        "mean_orig_val": (sum_orig_val / count_orig) if count_orig else None,
        "mean_orig_test": (sum_orig_test / count_orig) if count_orig else None,
        "mean_strip_val": (sum_strip_val / count_strip) if count_strip else None,
        "mean_strip_test": (sum_strip_test / count_strip) if count_strip else None,
        "examples_orig_vs_strip_diff": diff_examples,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
