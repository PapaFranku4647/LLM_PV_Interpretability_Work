from __future__ import annotations

import ast
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

from code_normalizer import sanitize_generated_code


Judgement = Literal["pass", "fail", "uncertain"]


CODE1_WRITER_TEMPLATE = """
You are given a thesis produced by a model.

Thesis conditions (raw text):
[CONDITIONS]

Thesis label:
[LABEL]

Reference sample format:
[SAMPLE]

Task:
Write Python code for a function `check_conditions(x)` that returns True
iff sample x satisfies the thesis conditions, and False otherwise.

Rules:
- Do NOT perform classification.
- Do NOT use imports.
- No side effects or I/O.
- Define exactly one top-level function: `def check_conditions(x):`.
- Avoid loops or comprehensions.
- Do NOT use classes, decorators, or async features.
- Return only bool/boolish values (`True/False` or equivalent).
- x may be either:
  1) dict with keys like x0, x1, ...
  2) list/tuple where xN maps to x[N]

Output STRICT JSON only:
{
  "code1": "def check_conditions(x):\\n    ..."
}
""".strip()


CODE1_VERIFIER_TEMPLATE = """
You must verify whether the candidate function `check_conditions(x)` matches
the thesis conditions.

Thesis conditions (raw text):
[CONDITIONS]

Thesis label:
[LABEL]

Candidate Code1:
```python
[CODE1]
```

Output STRICT JSON only:
{
  "judgement": "pass|fail|uncertain",
  "reason": "short explanation",
  "testcases": [
    {"sample": {"x0": 1, "x1": 5}, "expected": true, "note": "satisfies thesis"},
    {"sample": {"x0": 0, "x1": 1}, "expected": false, "note": "violates thesis"}
  ]
}

Testcase requirements:
- Provide thesis-grounded cases.
- Include both positive and negative cases.
- Prefer at least 3 positive and 3 negative cases when possible.
""".strip()


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
    "tuple": tuple,
}

_DISALLOWED_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "help",
}

_DISALLOWED_ATTR_NAMES = {
    "system",
    "popen",
    "spawn",
    "fork",
    "kill",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "makedirs",
    "read",
    "write",
    "send",
    "recv",
    "connect",
    "request",
    "urlopen",
}

_DISALLOWED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Delete,
)

_DISALLOWED_LOOP_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
)


@dataclass
class Code1GenerationResult:
    code1: Optional[str]
    compile_ok: bool
    compile_error: Optional[str]


@dataclass
class TestCase:
    sample: Any
    expected: bool
    note: str = ""


@dataclass
class SemanticVerificationResult:
    judgement: Judgement
    reason: str
    testcases: list[TestCase] = field(default_factory=list)


@dataclass
class TestcaseVerificationResult:
    total: int
    passed: int
    failed: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Code1VerificationBundle:
    final_code1: Optional[str]
    accepted: bool
    attempts: int
    semantic_result: Optional[SemanticVerificationResult]
    testcase_result: Optional[TestcaseVerificationResult]
    error: Optional[str]


def _obj_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_text_from_response(response: Any) -> str:
    direct = _obj_get(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = _obj_get(response, "output") or []
    for item in output:
        content = _obj_get(item, "content") or []
        if isinstance(content, list):
            for chunk in content:
                chunk_type = _obj_get(chunk, "type")
                if chunk_type in {"output_text", "text"}:
                    text_val = _obj_get(chunk, "text")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val.strip())
                    elif isinstance(text_val, dict):
                        nested = text_val.get("value")
                        if isinstance(nested, str) and nested.strip():
                            parts.append(nested.strip())
    return "\n".join(parts).strip()


def _parse_json_dict(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty_response"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_not_object"
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed, None
            return None, "fenced_json_not_object"
        except Exception as e:
            return None, f"fenced_json_parse_error: {e}"

    any_obj = re.search(r"\{[\s\S]*\}", text)
    if any_obj:
        try:
            parsed = json.loads(any_obj.group(0))
            if isinstance(parsed, dict):
                return parsed, None
            return None, "extracted_json_not_object"
        except Exception as e:
            return None, f"json_parse_error: {e}"
    return None, "no_json_object_found"


def _request_json_object(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int = 1200,
    reasoning_effort: str = "minimal",
    text_verbosity: str = "low",
) -> tuple[Optional[dict[str, Any]], Optional[str], str]:
    body = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": text_verbosity},
        "max_output_tokens": max_output_tokens,
        "tool_choice": "none",
    }
    response = client.responses.create(**body)
    response_text = _extract_text_from_response(response)
    parsed, parse_error = _parse_json_dict(response_text)
    return parsed, parse_error, response_text


def _build_code1_writer_prompt(
    thesis_conditions: str,
    thesis_label: int,
    sample_repr: str,
    feedback: str = "",
) -> str:
    prompt = CODE1_WRITER_TEMPLATE
    prompt = prompt.replace("[CONDITIONS]", str(thesis_conditions).strip())
    prompt = prompt.replace("[LABEL]", str(int(thesis_label)))
    prompt = prompt.replace("[SAMPLE]", str(sample_repr).strip())
    if feedback:
        prompt += f"\n\nPrevious attempt feedback:\n{feedback.strip()}\n"
    return prompt


def _build_code1_verifier_prompt(
    thesis_conditions: str,
    thesis_label: int,
    code1: str,
) -> str:
    prompt = CODE1_VERIFIER_TEMPLATE
    prompt = prompt.replace("[CONDITIONS]", str(thesis_conditions).strip())
    prompt = prompt.replace("[LABEL]", str(int(thesis_label)))
    prompt = prompt.replace("[CODE1]", (code1 or "").strip())
    return prompt


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "y"}:
            return True
        if lowered in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot coerce to bool: {value!r}")


def _normalize_testcases(raw: Any) -> tuple[list[TestCase], list[str]]:
    errors: list[str] = []
    if not isinstance(raw, list):
        return [], ["testcases_not_list"]

    testcases: list[TestCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"testcase_{idx}_not_object")
            continue
        sample = item.get("sample")
        if not isinstance(sample, (dict, list, tuple)):
            errors.append(f"testcase_{idx}_invalid_sample")
            continue
        try:
            expected = _coerce_bool(item.get("expected"))
        except Exception:
            errors.append(f"testcase_{idx}_invalid_expected")
            continue
        note = str(item.get("note", "")).strip()
        testcases.append(TestCase(sample=sample, expected=expected, note=note))
    return testcases, errors


def _validate_code1_ast(tree: ast.AST) -> Optional[str]:
    if not isinstance(tree, ast.Module):
        return "not_module_ast"
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        return "top_level_must_contain_only_function_defs"
    if len(tree.body) != 1:
        return "must_define_exactly_one_function"

    fn_node = tree.body[0]
    if fn_node.name != "check_conditions":
        return "function_name_must_be_check_conditions"
    if fn_node.decorator_list:
        return "decorators_not_allowed"
    if fn_node.args.vararg or fn_node.args.kwarg or fn_node.args.kwonlyargs:
        return "unsupported_function_signature"
    if len(fn_node.args.args) != 1 or fn_node.args.args[0].arg != "x":
        return "signature_must_be_check_conditions_x"
    if not any(isinstance(n, ast.Return) for n in ast.walk(fn_node)):
        return "check_conditions_must_return_boolish"

    for node in ast.walk(fn_node):
        if isinstance(node, _DISALLOWED_NODE_TYPES):
            return f"disallowed_node_type: {type(node).__name__}"
        if isinstance(node, _DISALLOWED_LOOP_TYPES):
            return f"disallowed_loop_or_comprehension: {type(node).__name__}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return "dunder_attribute_access_not_allowed"
            if node.attr in _DISALLOWED_ATTR_NAMES:
                return f"disallowed_attribute_call: {node.attr}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "dunder_name_not_allowed"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DISALLOWED_CALL_NAMES:
                return f"disallowed_call: {func.id}"
            if isinstance(func, ast.Attribute) and func.attr in _DISALLOWED_ATTR_NAMES:
                return f"disallowed_attribute_call: {func.attr}"
    return None


def compile_code1(code1: str) -> tuple[Optional[Callable[[Any], bool]], Optional[str]]:
    prepared = sanitize_generated_code(code1 or "")
    if not prepared.strip():
        return None, "empty_code1"

    try:
        tree = ast.parse(prepared)
    except Exception as e:
        return None, f"parse_error: {e}"

    validation_error = _validate_code1_ast(tree)
    if validation_error:
        return None, validation_error

    local_ns: dict[str, Any] = {}
    safe_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
    try:
        exec(compile(tree, filename="<code1>", mode="exec"), safe_globals, local_ns)
    except Exception as e:
        return None, f"exec_error: {e}"

    fn = local_ns.get("check_conditions")
    if not callable(fn):
        return None, "check_conditions_not_callable"
    return fn, None


def _run_with_timeout(
    fn: Callable[[Any], Any],
    sample: Any,
    timeout_s: float,
) -> tuple[Optional[Any], Optional[str]]:
    result: dict[str, Any] = {}
    error: dict[str, str] = {}

    def _target() -> None:
        try:
            result["value"] = fn(sample)
        except Exception as e:
            error["value"] = str(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(max(0.01, float(timeout_s)))

    if thread.is_alive():
        return None, "timeout"
    if "value" in error:
        return None, error["value"]
    return result.get("value"), None


def generate_code1_from_thesis(
    client: Any,
    model: str,
    thesis_conditions: str,
    thesis_label: int,
    sample_repr: str,
    *,
    max_output_tokens: int = 1200,
    reasoning_effort: str = "minimal",
    text_verbosity: str = "low",
    feedback: str = "",
) -> Code1GenerationResult:
    prompt = _build_code1_writer_prompt(
        thesis_conditions=thesis_conditions,
        thesis_label=thesis_label,
        sample_repr=sample_repr,
        feedback=feedback,
    )
    parsed, parse_error, _ = _request_json_object(
        client=client,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )
    if not isinstance(parsed, dict):
        return Code1GenerationResult(
            code1=None,
            compile_ok=False,
            compile_error=f"writer_json_parse_error: {parse_error}",
        )

    raw_code = parsed.get("code1")
    if not isinstance(raw_code, str) or not raw_code.strip():
        return Code1GenerationResult(
            code1=None,
            compile_ok=False,
            compile_error="writer_missing_code1_field",
        )

    code1 = sanitize_generated_code(raw_code)
    _, compile_error = compile_code1(code1)
    return Code1GenerationResult(
        code1=code1,
        compile_ok=compile_error is None,
        compile_error=compile_error,
    )


def verify_code1_semantics(
    client: Any,
    verifier_model: str,
    thesis_conditions: str,
    thesis_label: int,
    code1: str,
    *,
    max_output_tokens: int = 1200,
    reasoning_effort: str = "minimal",
    text_verbosity: str = "low",
) -> SemanticVerificationResult:
    prompt = _build_code1_verifier_prompt(
        thesis_conditions=thesis_conditions,
        thesis_label=thesis_label,
        code1=code1,
    )
    parsed, parse_error, _ = _request_json_object(
        client=client,
        model=verifier_model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )
    if not isinstance(parsed, dict):
        return SemanticVerificationResult(
            judgement="uncertain",
            reason=f"verifier_json_parse_error: {parse_error}",
            testcases=[],
        )

    raw_judgement = str(parsed.get("judgement", "")).strip().lower()
    if raw_judgement not in {"pass", "fail", "uncertain"}:
        judgement: Judgement = "uncertain"
    else:
        judgement = raw_judgement  # type: ignore[assignment]

    reason = str(parsed.get("reason", "")).strip() or "no_reason_provided"
    testcases, testcase_errors = _normalize_testcases(parsed.get("testcases"))
    if testcase_errors:
        reason = f"{reason} | testcase_parse_errors: {', '.join(testcase_errors)}"
    return SemanticVerificationResult(judgement=judgement, reason=reason, testcases=testcases)


def verify_code1_with_testcases(
    code1_callable: Callable[[Any], Any],
    testcases: Sequence[TestCase],
    *,
    execution_timeout_s: float = 1.0,
) -> TestcaseVerificationResult:
    mismatches: list[dict[str, Any]] = []
    total = len(testcases)

    for idx, testcase in enumerate(testcases):
        raw_out, call_error = _run_with_timeout(code1_callable, testcase.sample, execution_timeout_s)
        if call_error is not None:
            mismatches.append(
                {
                    "index": idx,
                    "sample": testcase.sample,
                    "expected": testcase.expected,
                    "actual": None,
                    "error": call_error,
                    "note": testcase.note,
                }
            )
            continue

        try:
            actual = _coerce_bool(raw_out)
        except Exception as e:
            mismatches.append(
                {
                    "index": idx,
                    "sample": testcase.sample,
                    "expected": testcase.expected,
                    "actual": raw_out,
                    "error": f"non_boolish_return: {e}",
                    "note": testcase.note,
                }
            )
            continue

        if actual != testcase.expected:
            mismatches.append(
                {
                    "index": idx,
                    "sample": testcase.sample,
                    "expected": testcase.expected,
                    "actual": actual,
                    "error": "expected_mismatch",
                    "note": testcase.note,
                }
            )

    failed = len(mismatches)
    passed = total - failed
    return TestcaseVerificationResult(total=total, passed=passed, failed=failed, mismatches=mismatches)


def _build_retry_feedback(
    generation_error: Optional[str],
    semantic_result: Optional[SemanticVerificationResult],
    testcase_result: Optional[TestcaseVerificationResult],
) -> str:
    parts: list[str] = []
    if generation_error:
        parts.append(f"Generation/compile error: {generation_error}")
    if semantic_result is not None:
        parts.append(f"Verifier judgement: {semantic_result.judgement}")
        parts.append(f"Verifier reason: {semantic_result.reason}")
    if testcase_result is not None:
        parts.append(
            f"Testcase results: passed={testcase_result.passed}, failed={testcase_result.failed}, total={testcase_result.total}"
        )
        if testcase_result.mismatches:
            sample_mismatch = testcase_result.mismatches[0]
            parts.append(f"First mismatch example: {json.dumps(sample_mismatch, ensure_ascii=False)}")
    return "\n".join(parts)


def _has_positive_and_negative_testcases(testcases: Sequence[TestCase]) -> bool:
    if not testcases:
        return False
    has_positive = any(tc.expected for tc in testcases)
    has_negative = any(not tc.expected for tc in testcases)
    return has_positive and has_negative


def build_code1_with_verification(
    client: Any,
    writer_model: str,
    verifier_model: str,
    thesis_conditions: str,
    thesis_label: int,
    sample_repr: str,
    *,
    retry_once: bool = True,
    max_output_tokens: int = 1200,
    reasoning_effort: str = "minimal",
    text_verbosity: str = "low",
    execution_timeout_s: float = 1.0,
) -> Code1VerificationBundle:
    max_attempts = 2 if retry_once else 1
    feedback = ""

    last_semantic: Optional[SemanticVerificationResult] = None
    last_testcase: Optional[TestcaseVerificationResult] = None
    final_code1: Optional[str] = None
    final_error: Optional[str] = None

    for attempt_idx in range(1, max_attempts + 1):
        generation = generate_code1_from_thesis(
            client=client,
            model=writer_model,
            thesis_conditions=thesis_conditions,
            thesis_label=thesis_label,
            sample_repr=sample_repr,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            feedback=feedback,
        )
        final_code1 = generation.code1

        if not generation.compile_ok or not generation.code1:
            final_error = generation.compile_error or "code1_generation_failed"
            feedback = _build_retry_feedback(final_error, None, None)
            continue

        code1_callable, compile_error = compile_code1(generation.code1)
        if code1_callable is None:
            final_error = compile_error or "code1_compile_failed"
            feedback = _build_retry_feedback(final_error, None, None)
            continue

        semantic = verify_code1_semantics(
            client=client,
            verifier_model=verifier_model,
            thesis_conditions=thesis_conditions,
            thesis_label=thesis_label,
            code1=generation.code1,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        last_semantic = semantic

        testcase_result = verify_code1_with_testcases(
            code1_callable=code1_callable,
            testcases=semantic.testcases,
            execution_timeout_s=execution_timeout_s,
        )
        last_testcase = testcase_result

        testcase_balance_ok = _has_positive_and_negative_testcases(semantic.testcases)
        semantic_pass = semantic.judgement == "pass"
        testcase_pass = testcase_result.total > 0 and testcase_result.failed == 0 and testcase_balance_ok

        if semantic_pass and testcase_pass:
            return Code1VerificationBundle(
                final_code1=generation.code1,
                accepted=True,
                attempts=attempt_idx,
                semantic_result=semantic,
                testcase_result=testcase_result,
                error=None,
            )

        failure_reasons = []
        if not semantic_pass:
            failure_reasons.append(f"semantic_judgement={semantic.judgement}")
        if testcase_result.total == 0:
            failure_reasons.append("no_testcases")
        if testcase_result.failed > 0:
            failure_reasons.append(f"testcase_failed={testcase_result.failed}")
        if not testcase_balance_ok:
            failure_reasons.append("missing_positive_or_negative_testcases")
        final_error = "; ".join(failure_reasons) or "verification_failed"
        feedback = _build_retry_feedback(final_error, semantic, testcase_result)

    return Code1VerificationBundle(
        final_code1=final_code1,
        accepted=False,
        attempts=max_attempts,
        semantic_result=last_semantic,
        testcase_result=last_testcase,
        error=final_error or "verification_failed_after_retries",
    )
