# =====================================================================================
# runner.py — COMMENTED VERSION
# =====================================================================================
# High‑level purpose
# ------------------
# This script orchestrates an end‑to‑end experiment loop where an OpenAI GPT-5
# model is prompted with synthetic classification data and asked to output a
# Python function that reproduces the target mapping. The code:
#   • Builds persistent train/val/test splits per (target_function, sequence_length).
#   • Calls the OpenAI Responses API asynchronously with concurrency control.
#   • Extracts a candidate function from the model's JSON output, compiles it,
#     and evaluates its accuracy on validation (and then test, on perfect val).
#   • Performs multiple attempts per grid point and supports early-stopping when 
#     validation accuracy hits 1.0. If no attempt achieves perfect validation, 
#     all results are logged for post-processing to select the best one.
#     can choose the best validation in postprocessing). 
#   • Logs all steps as structured JSON to stdout and to runner.log.
#   • Emits a JSONL and CSV with detailed metrics per attempt.
#
# Some more technical details:
#   1) Dataset persistence & determinism: For any (fn, L) pair and a global seed,
#      derived_seed = (hash((fn, L)) & 0x7fffffff) ^ global_seed. Splits are saved
#      under datasets/<target>/L<length>/seed<derived_seed>/, so subsequent runs
#      reuse identical data, making results reproducible.
#   2) Prompts demand a single JSON object with the generated function under
#      the "code" key, simplifying parsing.
#   3) Compilation sandbox: we `exec` inside a controlled namespace and pick the
#      function named `f` if present (else the first def). We normalize predictions
#      to {0,1} via _normalize_pred_to01 (supports ints/bools/strings/tensor scalars).
#   4) Early stop: if validation accuracy is exactly 1.0, we compute test and stop.
# =====================================================================================

from __future__ import annotations
import os, sys, json, csv, time, argparse, asyncio, re, ast, random, tempfile, shutil, hashlib
import httpx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Mapping, Callable, Sequence
import logging

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.data_handler import get_data_generator, create_stratified_splits
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA
from prompt_variants import get_prompt_variant_suffix
from code_normalizer import normalize_generated_code, sanitize_generated_code
from llm_client import (
    build_async_client,
    build_chat_completions_body,
    call_llm_async,
    estimate_usage_cost as shared_estimate_usage_cost,
    flatten_cost_fields as shared_flatten_cost_fields,
    infer_default_api_mode,
    resolve_api_key_from_env,
    resolve_api_version_from_env,
    resolve_azure_endpoint_from_env,
    resolve_default_model_from_env,
)

external_get_accuracy = None


# =========================
# Usage normalization
# =========================

def normalize_usage(usage_obj) -> Dict[str, Any]:
    if not usage_obj:
        return {}
    if isinstance(usage_obj, Mapping):
        u = dict(usage_obj)
    else:
        u = {}
        for k in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_tokens", "output_tokens", "reasoning_tokens"
        ):
            v = getattr(usage_obj, k, None)
            if v is not None:
                u[k] = v

        def _to_dict(obj):
            if obj is None:
                return None
            if isinstance(obj, Mapping):
                return dict(obj)
            d = {}
            for kk in ("cached_tokens", "audio_tokens", "reasoning_tokens"):
                vv = getattr(obj, kk, None)
                if vv is not None:
                    d[kk] = vv
            return d or None

        ptd = getattr(usage_obj, "prompt_tokens_details", None)
        itd = getattr(usage_obj, "input_token_details", None)
        otd = getattr(usage_obj, "output_tokens_details", None)
        ctd = getattr(usage_obj, "completion_tokens_details", None)
        if (ptd := _to_dict(ptd)) is not None:
            u["prompt_tokens_details"] = ptd
        if (itd := _to_dict(itd)) is not None:
            u["input_token_details"] = itd
        if (otd := _to_dict(otd)) is not None:
            u["output_tokens_details"] = otd
        if (ctd := _to_dict(ctd)) is not None:
            u["completion_tokens_details"] = ctd

    if "prompt_tokens" not in u and "input_tokens" in u:
        u["prompt_tokens"] = u["input_tokens"]
    if "completion_tokens" not in u and "output_tokens" in u:
        u["completion_tokens"] = u["output_tokens"]

    details = u.get("prompt_tokens_details") or u.get("input_token_details") or {}
    if "cached_tokens" in details:
        u["cached_tokens"] = details["cached_tokens"]

    if "reasoning_tokens" not in u or u.get("reasoning_tokens") is None:
        rt = None
        for dkey in ("output_tokens_details", "completion_tokens_details", "input_token_details"):
            d = u.get(dkey) or {}
            if isinstance(d, Mapping):
                rt = d.get("reasoning_tokens")
            if rt is not None:
                break
        if rt is not None:
            u["reasoning_tokens"] = rt

    return u


# =========================
# Logging
# =========================

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
    logger = logging.getLogger("runner")
    logger.setLevel(level.upper())

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())

    file_handler = logging.FileHandler("program_synthesis/runner.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    logger.handlers[:] = [stream_handler, file_handler]
    logger.propagate = False
    return logger


# =========================
# Config
# =========================

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", ""}


@dataclass
class Config:
    api_key: str = field(default_factory=resolve_api_key_from_env)
    api_base_url: str = os.getenv("API_BASE_URL", "").strip()
    azure_endpoint: str = field(default_factory=resolve_azure_endpoint_from_env)
    api_version: str = field(default_factory=resolve_api_version_from_env)
    api_mode: str = field(default_factory=infer_default_api_mode)
    model: str = field(default_factory=lambda: resolve_default_model_from_env("gpt-5"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "20000"))
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "high")
    verbosity: Optional[str] = os.getenv("TEXT_VERBOSITY", "low")
    tool_choice: str = os.getenv("TOOL_CHOICE", "auto")
    sanitize_generated_code: bool = field(default_factory=lambda: _env_flag("SANITIZE_GENERATED_CODE", True))
    prompt_variant: str = os.getenv("PROMPT_VARIANT", "standard").lower()
    enable_code_interpreter: bool = os.getenv("ENABLE_CODE_INTERPRETER", "0") == "1"
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "0") == "1"
    code0_train_mode: str = os.getenv("CODE0_TRAIN_MODE", "normal").strip().lower()
    code0_batch_size: int = int(os.getenv("CODE0_BATCH_SIZE", "0"))

    dry_run: bool = os.getenv("DRY_RUN", "0") == "1"
    concurrency: int = int(os.getenv("CONCURRENCY", "5"))
    per_call_timeout_s: float = float(os.getenv("PER_CALL_TIMEOUT_S", "1200"))
    retry_delay_s: float = float(os.getenv("RETRY_DELAY_S", "5"))

    functions: List[str] = field(default_factory=lambda: [
        "fn_a", "fn_b", "fn_c", "fn_d", "fn_e", "fn_f",
        "fn_g", "fn_h", "fn_i", "fn_j", "fn_k", "fn_l", "fn_v", "fn_t",
        "fn_m", "fn_n", "fn_o", "fn_p", "fn_q", "fn_aa",
    ])
    lengths: List[int] = field(default_factory=lambda: [100, 50, 30, 25, 20])
    attempts: int = int(os.getenv("ATTEMPTS", "5"))
    num_trials: int = int(os.getenv("NUM_TRIALS", "5"))

    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int   = int(os.getenv("VAL_SIZE",   "100"))
    test_size: int  = int(os.getenv("TEST_SIZE",  "10000"))
    seed: int       = int(os.getenv("GLOBAL_SEED","42"))
    dataset_dir: str = os.getenv("DATASET_DIR", "program_synthesis/datasets")

    out_jsonl: str = os.getenv("OUT_JSONL", "program_synthesis/results_attempts.jsonl")
    out_csv: str   = os.getenv("OUT_CSV",   "program_synthesis/results_attempts.csv")
    out_manifest: str = os.getenv("OUT_MANIFEST", "")
    run_id: str = field(default_factory=lambda: os.getenv("RUN_ID", time.strftime("%Y%m%d_%H%M%S")))


# =========================
# Prompt
# =========================

def build_user_prompt(
    data_examples: List[str],
    seq_len: int,
    decimal: bool = False,
    tabular: bool = False,
    prompt_variant: str = "standard",
) -> str:
    if tabular:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"Given tabular input data (comma-separated feature:value pairs) and their corresponding scalar binary outputs ('0' or '1'), "
            f"find a concise Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The argument `x` is a Python dict mapping feature names to values (e.g. x['x3'] gives a float, x['x0'] gives a category string like 'c1'). "
            f"The function should not be a trainable model, but a direct logical or mathematical representation of the target function."
        )
    elif decimal:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"Given a sequence of input vectors (decimal, length {seq_len}) and their corresponding scalar binary outputs ('0' or '1'), "
            f"find a concise Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or mathematical representation of the target function."
        )
    else:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"Given a sequence of input vectors (binary, length {seq_len}) and their corresponding scalar binary outputs ('0' or '1'), "
            f"find a concise Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or mathematical representation of the target function."
        )
    prompt = f"{problem_statement}\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += 'You must output ONLY a single JSON object: {"code": "<python function>"}. Keep the function concise (under 30 lines, no comments). Output the JSON immediately with no other text.'
    variant_suffix = get_prompt_variant_suffix(prompt_variant)
    if variant_suffix:
        prompt += "\n\n" + variant_suffix
    return prompt


def find_misclassified_examples(
    code_str: str,
    data_lines: List[str],
    is_tabular: bool = False,
    max_examples: int = 10,
    sanitize: bool = True,
) -> List[str]:
    """Return up to max_examples lines that the code misclassifies."""
    if not code_str or not data_lines:
        return []
    try:
        fn_callable = compile_callable_from_code(code_str, sanitize=sanitize)
    except Exception:
        return []
    misclassified = []
    for line in data_lines:
        try:
            x, y = line.split("->")
            x = x.strip()
            y_int = int(y.strip())
            if is_tabular:
                x = _parse_tabular_input(x)
            pred = fn_callable(x)
            pred_int = _normalize_pred_to01(pred)
            if pred_int != y_int:
                misclassified.append(line.strip())
                if len(misclassified) >= max_examples:
                    break
        except Exception:
            continue
    return misclassified


def build_batch_revision_prompt(
    previous_code: str,
    data_examples: List[str],
    seq_len: int,
    decimal: bool = False,
    tabular: bool = False,
    prompt_variant: str = "standard",
    train_acc: Optional[float] = None,
    misclassified_examples: Optional[List[str]] = None,
) -> str:
    if tabular:
        data_type = "tabular input data (comma-separated feature:value pairs)"
        x_desc = "The argument `x` is a Python dict mapping feature names to values (e.g. x['x3'] gives a float, x['x0'] gives a category string like 'c1'). "
    elif decimal:
        data_type = f"decimal input vectors (length {seq_len})"
        x_desc = ""
    else:
        data_type = f"binary input vectors (length {seq_len})"
        x_desc = ""

    prompt = (
        f"**Problem Statement:**\n"
        f"You previously wrote a Python function `f(x)` for {data_type} "
        f"with scalar binary outputs ('0' or '1'). {x_desc}"
        f"Revise that function using the feedback and new training batch below.\n\n"
    )
    prompt += "**Current Best Function:**\n```python\n" + previous_code + "\n```\n\n"

    if train_acc is not None:
        prompt += f"**Performance:** This function achieves {train_acc:.0%} accuracy on all training data seen so far.\n\n"

    if misclassified_examples:
        prompt += "**Misclassified Examples** (your function gets these WRONG — fix them):\n```\n"
        prompt += "\n".join(misclassified_examples) + "\n```\n\n"

    prompt += "**New Batch Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += (
        "Revise the function to correctly handle both the misclassified examples above "
        "AND the new batch, while preserving correctness on previously seen data. "
        "Keep the function concise. Return the full revised function.\n"
    )
    prompt += 'You must output ONLY a single JSON object: {"code": "<python function>"}. Keep the function concise (under 30 lines, no comments). Output the JSON immediately with no other text.'
    variant_suffix = get_prompt_variant_suffix(prompt_variant)
    if variant_suffix:
        prompt += "\n\n" + variant_suffix
    return prompt


# =========================
# Function mapping & special sets
# =========================

FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING

DECIMAL_FNS = {"prime_decimal", "prime_decimal_tf_check", "prime_plus_47", "collatz_steps_parity"}
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}


# =========================
# Atomic file helpers
# =========================

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _safe_write_text_lines(path: str, lines: List[str]) -> None:
    _ensure_parent_dir(path)
    tmp_dir = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=tmp_dir) as tmp:
        for ln in lines:
            tmp.write(f"{ln}\n")
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    tmp_dir = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=tmp_dir) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


# =========================
# DatasetStore
# =========================

class DatasetStore:
    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _paths(self, target_name: str, L: int, derived_seed: int) -> Dict[str, str]:
        base = os.path.join(self.cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
        return {
            "dir": base,
            "train": os.path.join(base, "train.txt"),
            "val":   os.path.join(base, "val.txt"),
            "test":  os.path.join(base, "test.txt"),
            "meta":  os.path.join(base, "meta.json"),
        }

    def _generate_lines(self, target_name: str, L: int, size: int) -> List[str]:
        gen = get_data_generator(target_name, L, size)
        dataset = gen.generate_data()
        return [f"{''.join(sample['Input'])} -> {sample['Output']}" for sample in dataset]
    
    def _stable_derived_seed(self, fn: str, L: int) -> int:
        key = (
            f"{fn}|L={L}|train={self.cfg.train_size+self.cfg.val_size}|test={self.cfg.test_size}"
            f"|base_seed={self.cfg.seed}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return (int.from_bytes(digest[:8], "big") & 0x7FFFFFFF)

    def _ensure_splits(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool, bool]:
        target_name = FUNCTION_NAME_MAPPING[fn]
        is_decimal = target_name in DECIMAL_FNS
        is_tabular = target_name in TABULAR_FNS

        derived_seed = self._stable_derived_seed(fn, L)
        paths = self._paths(target_name, L, derived_seed)

        def exists_with_size(path: str, expect: int) -> bool:
            if not os.path.exists(path): return False
            try:
                return sum(1 for _ in open(path, "r", encoding="utf-8")) == expect
            except Exception:
                return False

        if (exists_with_size(paths["train"], self.cfg.train_size) and
            exists_with_size(paths["val"],   self.cfg.val_size)   and
            exists_with_size(paths["test"],  self.cfg.test_size)):
            self.log.info("dataset_reused", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
            return _read_lines(paths["train"]), _read_lines(paths["val"]), _read_lines(paths["test"]), is_decimal, is_tabular

        self._set_seed(derived_seed)
        self.log.info("dataset_generating", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})

        total_samples = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        generator = get_data_generator(target_name, L, total_samples)
        all_samples_dicts = generator.generate_data()

        train_split_dicts, val_split_dicts, test_split_dicts = create_stratified_splits(
            all_samples=all_samples_dicts,
            train_size=self.cfg.train_size,
            val_size=self.cfg.val_size,
            test_size=self.cfg.test_size,
            device='cpu'
        )

        train_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in train_split_dicts]
        val_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in val_split_dicts]
        test_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in test_split_dicts]
        
        random.shuffle(train_lines)
        random.shuffle(val_lines)
        
        _safe_write_text_lines(paths["train"], train_lines)
        _safe_write_text_lines(paths["val"],   val_lines)
        _safe_write_text_lines(paths["test"],  test_lines)
        _safe_write_json(paths["meta"], {
            "fn": fn, "target_name": target_name, "length": L, 
            "decimal": is_decimal, "tabular": is_tabular,
            "derived_seed": derived_seed,
            "sizes": {"train": self.cfg.train_size, "val": self.cfg.val_size, "test": self.cfg.test_size},
            "created_ts": int(time.time())
        })

        self.log.info("dataset_written", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
        return train_lines, val_lines, test_lines, is_decimal, is_tabular

    def get(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool, bool]:
        return self._ensure_splits(fn, L)

    def derived_seed(self, fn: str, L: int) -> int:
        return self._stable_derived_seed(fn, L)


# =========================
# Code extraction & compilation
# =========================

def extract_code_from_output(output_text: str) -> Optional[str]:
    if not output_text:
        return None
    # Strip <think>...</think> blocks (e.g., from Qwen, DeepSeek reasoning models)
    cleaned = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = output_text
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
            return obj["code"]
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
                return obj["code"]
        except Exception:
            return None
    return None


def _flatten_chat_content_text(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, Mapping):
        nested = content.get("text")
        if isinstance(nested, str):
            return [nested]
        return _flatten_chat_content_text(content.get("content"))
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            parts.extend(_flatten_chat_content_text(item))
        return parts
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return [text_attr]
    nested_attr = getattr(content, "content", None)
    if nested_attr is not None:
        return _flatten_chat_content_text(nested_attr)
    return []


def _extract_text_parts_from_chat_choice(choice: Any) -> List[str]:
    parts: List[str] = []
    if isinstance(choice, Mapping):
        delta = choice.get("delta")
        message = choice.get("message")
        text_value = choice.get("text")
    else:
        delta = getattr(choice, "delta", None)
        message = getattr(choice, "message", None)
        text_value = getattr(choice, "text", None)

    parts.extend(_flatten_chat_content_text(text_value))

    if delta is not None:
        if isinstance(delta, Mapping):
            parts.extend(_flatten_chat_content_text(delta.get("content")))
        else:
            parts.extend(_flatten_chat_content_text(getattr(delta, "content", None)))

    if message is not None:
        if isinstance(message, Mapping):
            parts.extend(_flatten_chat_content_text(message.get("content")))
        else:
            parts.extend(_flatten_chat_content_text(getattr(message, "content", None)))

    return [p for p in parts if isinstance(p, str) and p]


def extract_text_from_chat_completion(res: Any) -> str:
    if isinstance(res, str):
        return parse_chat_completion_sse(res).get("text", "")

    choices = getattr(res, "choices", None)
    if choices is None and isinstance(res, Mapping):
        choices = res.get("choices")
    choices = choices or []
    if not choices:
        return ""
    first = choices[0]
    direct_parts = _extract_text_parts_from_chat_choice(first)
    if direct_parts:
        return "".join(direct_parts).strip()
    msg = getattr(first, "message", None)
    if msg is None and isinstance(first, Mapping):
        msg = first.get("message")
    if msg is None:
        return ""

    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, Mapping):
        content = msg.get("content")

    return "".join(_flatten_chat_content_text(content)).strip()


def parse_chat_completion_sse(raw: str) -> Dict[str, Any]:
    text_parts: List[str] = []
    usage: Dict[str, Any] = {}
    model_name: Optional[str] = None
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("data:"):
            continue
        payload = ln[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        if isinstance(obj, Mapping):
            m = obj.get("model")
            if isinstance(m, str) and m:
                model_name = m

            u = obj.get("usage")
            if isinstance(u, Mapping):
                usage = dict(u)

            choices = obj.get("choices")
            if isinstance(choices, list):
                for c in choices:
                    if not isinstance(c, Mapping):
                        continue
                    text_parts.extend(_extract_text_parts_from_chat_choice(c))
    return {
        "text": "".join(text_parts).strip(),
        "usage": usage,
        "model": model_name,
    }


def parse_chat_completion_http_payload(payload_text: str, content_type: str = "") -> Dict[str, Any]:
    text = payload_text or ""
    lowered_content_type = (content_type or "").lower()
    if "text/event-stream" in lowered_content_type or text.lstrip().startswith("data:"):
        parsed = parse_chat_completion_sse(text)
        return {
            "text": parsed.get("text", ""),
            "usage": normalize_usage(parsed.get("usage") or {}),
            "raw_payload": text,
        }

    data = json.loads(text)
    usage = {}
    if isinstance(data, Mapping):
        usage = normalize_usage(data.get("usage") or {})
    return {
        "text": extract_text_from_chat_completion(data),
        "usage": usage,
        "raw_payload": text,
    }

def compile_callable_from_code(code_str: str, sanitize: bool = True) -> Callable[[str], int]:
    prepared_code = sanitize_generated_code(code_str) if sanitize else normalize_generated_code(code_str)
    tree = ast.parse(prepared_code)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found in generated code.")
    prefer_name = "f" if "f" in fn_names else fn_names[0]
    local_ns: Dict[str, Any] = {}
    safe_globals = {"__builtins__": __builtins__}
    exec(compile(tree, filename="<generated>", mode="exec"), safe_globals, local_ns)
    fn = local_ns.get(prefer_name)
    if not callable(fn):
        raise ValueError(f"Function '{prefer_name}' not found after exec.")
    return fn

def analyze_code_structure(code_str: str, sanitize: bool = True) -> Dict[str, Any]:
    normalized = sanitize_generated_code(code_str) if sanitize else normalize_generated_code(code_str)

    non_empty_lines = [ln for ln in normalized.splitlines() if ln.strip()]
    out = {
        "code": normalized,
        "code_lines": len(non_empty_lines),
        "num_branches": None,
        "code_analysis_error": None,
    }
    try:
        tree = ast.parse(normalized)
        out["num_branches"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
    except Exception as e:
        out["code_analysis_error"] = str(e)
    return out


# =========================
# Accuracy evaluation
# =========================

def _normalize_pred_to01(pred) -> int:
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

def _parse_tabular_input(x_str: str) -> Dict[str, Any]:
    result = {}
    for pair in x_str.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
    return result

def _local_get_accuracy(fn_callable: Callable[[str], int], data_lines: List[str], logger=None, is_tabular: bool=False) -> float:
    if not data_lines:
        return 0.0
    correct = 0
    errors = 0
    for line in data_lines:
        try:
            x, y = line.split("->")
            x = x.strip()
            y_int = int(y.strip())
            if is_tabular:
                x = _parse_tabular_input(x)
            pred = fn_callable(x)
            pred_int = _normalize_pred_to01(pred)
            correct += int(pred_int == y_int)
        except Exception as e:
            if errors < 5:
                logger.debug("Evaluation error on generated code", extra={"line": line, "error": str(e)})
            errors += 1
    if errors > 0:
        logger.warning(f"Encountered {errors} errors during evaluation of generated code.")
    return correct / len(data_lines)

def evaluate_accuracy(fn_callable: Callable[[str], int], data_lines: List[str], logger=None, is_tabular: bool=False) -> float:
    if external_get_accuracy is not None:
        try:
            return float(external_get_accuracy(fn_callable, data_lines))
        except Exception:
            pass
    return _local_get_accuracy(fn_callable, data_lines, logger, is_tabular)


def should_update_best_by_val(current_val_acc: Optional[float], best_val_acc: Optional[float]) -> bool:
    if current_val_acc is None:
        return False
    if best_val_acc is None:
        return True
    return current_val_acc > best_val_acc


def should_update_best_by_train(
    current_train_acc: Optional[float],
    best_train_acc: Optional[float],
    current_batch_train_acc: Optional[float],
    best_batch_train_acc: Optional[float],
    current_val_acc: Optional[float],
    best_val_acc: Optional[float],
) -> bool:
    if current_train_acc is None:
        return False
    if best_train_acc is None:
        return True
    if current_train_acc > best_train_acc:
        return True
    if current_train_acc < best_train_acc:
        return False
    if current_batch_train_acc is not None and best_batch_train_acc is None:
        return True
    if current_batch_train_acc is None and best_batch_train_acc is not None:
        return False
    if current_batch_train_acc is not None and best_batch_train_acc is not None:
        if current_batch_train_acc > best_batch_train_acc:
            return True
        if current_batch_train_acc < best_batch_train_acc:
            return False
    return should_update_best_by_val(current_val_acc, best_val_acc)


def split_train_batches(train_lines: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [train_lines[i:i + batch_size] for i in range(0, len(train_lines), batch_size)]


def derive_batch_artifact_paths(cfg: Config) -> Tuple[str, str]:
    batch_jsonl = os.path.splitext(cfg.out_jsonl)[0] + "_batches.jsonl"
    batch_csv = os.path.splitext(cfg.out_csv)[0] + "_batches.csv"
    return batch_jsonl, batch_csv


def chat_completion_supports_reasoning_effort(model_name: str) -> bool:
    model_l = (model_name or "").strip().lower()
    return (
        model_l.startswith("o3")
        or model_l.startswith("o4")
        or "gpt-5" in model_l
        or model_l.startswith("openai/o3")
        or model_l.startswith("openai/o4")
    )

def build_row_run_metadata(cfg: Config) -> Dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "api_mode": cfg.api_mode,
        "api_base_url": cfg.api_base_url,
        "azure_endpoint": cfg.azure_endpoint,
        "api_version": cfg.api_version,
        "model": cfg.model,
        "reasoning_effort": cfg.reasoning_effort,
        "max_output_tokens": cfg.max_output_tokens,
        "tool_choice": cfg.tool_choice,
        "prompt_variant": cfg.prompt_variant,
        "enable_code_interpreter": cfg.enable_code_interpreter,
        "global_seed": cfg.seed,
        "train_size": cfg.train_size,
        "val_size": cfg.val_size,
        "test_size": cfg.test_size,
        "dataset_dir": cfg.dataset_dir,
        "attempts_requested": cfg.attempts,
        "num_trials_requested": cfg.num_trials,
        "dry_run": cfg.dry_run,
        "retry_delay_s": cfg.retry_delay_s,
        "code0_train_mode": cfg.code0_train_mode,
        "code0_batch_size": cfg.code0_batch_size if cfg.code0_train_mode == "batched" else None,
    }


# =========================
# Runner
# =========================

class Runner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        if not cfg.api_key:
            raise SystemExit("API key is required. Set TAMU_API_KEY, TAMUS_AI_CHAT_API_KEY, dpf-key, or OPENAI_API_KEY.")
        if cfg.api_mode not in {"responses", "chat_completions"}:
            raise SystemExit("API_MODE must be 'responses' or 'chat_completions'.")
        self.cfg = cfg
        self.log = logger
        self.client, self.client_type = build_async_client(
            cfg.api_key,
            api_base_url=cfg.api_base_url,
            azure_endpoint=cfg.azure_endpoint,
            api_version=cfg.api_version,
        )
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.row_meta = {
            **build_row_run_metadata(cfg),
            "client_type": self.client_type,
        }

        self.tools: List[Dict[str, Any]] = []
        if cfg.enable_code_interpreter:
            self.tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

        self.ds = DatasetStore(cfg, logger)
        self.batch_winner_rows: List[Dict[str, Any]] = []

    async def _call_chat_completions_raw(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.api_base_url:
            raise RuntimeError("Raw chat completions require api_base_url.")
        url = self.cfg.api_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.cfg.per_call_timeout_s) as client:
            try:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                parsed = parse_chat_completion_http_payload(
                    response.text,
                    response.headers.get("content-type", ""),
                )
                if parsed.get("text") or body.get("stream") is True:
                    return parsed
                content_type = (response.headers.get("content-type", "") or "").lower()
                if "text/event-stream" not in content_type:
                    return parsed
            except httpx.HTTPStatusError:
                if body.get("stream") is True:
                    raise

            retry_body = dict(body)
            retry_body["stream"] = True
            retry_body["stream_options"] = {"include_usage": True}
            async with client.stream("POST", url, headers=headers, json=retry_body) as response:
                response.raise_for_status()
                chunks: List[str] = []
                async for chunk in response.aiter_text():
                    chunks.append(chunk)
                return parse_chat_completion_http_payload(
                    "".join(chunks),
                    response.headers.get("content-type", ""),
                )

    def _attach_meta(self, row: Dict[str, Any], dataset_seed: Optional[int] = None) -> Dict[str, Any]:
        out = {**self.row_meta, **row}
        if dataset_seed is not None:
            out["dataset_seed"] = dataset_seed
        return out

    async def _call_with_prompt(self, fn: str, L: int, attempt_idx: int, prompt_text: str) -> Dict[str, Any]:
        if self.cfg.api_mode == "chat_completions":
            msg_payload = [{"role": "user", "content": prompt_text}]
            body_preview_size = len(json.dumps({"messages": msg_payload}))
            body = build_chat_completions_body(
                model=self.cfg.model,
                messages=msg_payload,
                max_output_tokens=self.cfg.max_output_tokens,
                reasoning_effort=self.cfg.reasoning_effort,
                stream=False,
                enable_thinking=self.cfg.enable_thinking,
                api_base_url=self.cfg.api_base_url,
                azure_endpoint=self.cfg.azure_endpoint,
            )
        else:
            body_preview_size = len(json.dumps({"input":[{"role":"user","content":[{"type":"input_text","text": prompt_text}]}]}))
            body = {
                "model": self.cfg.model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
                "reasoning": {"effort": self.cfg.reasoning_effort},
                "max_output_tokens": self.cfg.max_output_tokens,
                "tool_choice": self.cfg.tool_choice,
            }
            if self.tools:
                body["tools"] = self.tools
            if self.cfg.verbosity:
                body["text"] = {"verbosity": self.cfg.verbosity}

        if self.cfg.dry_run:
            self.log.info("dry_run_input", extra={"fn": fn, "length": L, "attempt": attempt_idx, "prompt_preview": prompt_text})
            return {
                "fn": fn, "length": L, "attempt": attempt_idx,
                "prompt": prompt_text, "request_body": body,
                "text": None, "usage": {}, "cost": {}, "cached_tokens": 0, "duration_ms": 0
            }

        async def _try_call(tag: str):
            t0 = time.perf_counter()
            async with self.sem:
                llm_result = await asyncio.wait_for(
                    call_llm_async(
                        client=self.client,
                        api_mode=self.cfg.api_mode,
                        body=body,
                        timeout=self.cfg.per_call_timeout_s,
                        api_base_url=self.cfg.api_base_url,
                        api_key=self.cfg.api_key,
                        azure_endpoint=self.cfg.azure_endpoint,
                    ),
                    timeout=self.cfg.per_call_timeout_s,
                )
                out_text = llm_result.text
                usage = normalize_usage(llm_result.usage)
                tool_uses = llm_result.tool_uses
                tool_results_chars = llm_result.tool_results_chars
                returned_model = llm_result.model or self.cfg.model
                cost = shared_estimate_usage_cost(usage, returned_model)

            dt_ms = int((time.perf_counter() - t0) * 1000)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            self.log.info(tag, extra={
                "fn": fn, "length": L, "attempt": attempt_idx,
                "duration_ms": dt_ms, "prompt_chars": len(prompt_text),
                "request_body_bytes": len(json.dumps(body)),
                "input_section_bytes": body_preview_size,
                "tools_enabled": bool(self.tools), "tool_count": len(self.tools),
                "tool_uses": tool_uses, "tool_results_chars": tool_results_chars,
                "prompt_tokens": usage.get("prompt_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                "cached_tokens": cached, "completion_tokens": usage.get("completion_tokens"),
                "estimated_total_cost_usd": cost.get("estimated_total_cost_usd"),
            })
            return {
                "fn": fn, "length": L, "attempt": attempt_idx,
                "prompt": prompt_text,
                "text": out_text, "usage": usage, "cost": cost, "returned_model": returned_model,
                "cached_tokens": cached, "duration_ms": dt_ms,
                "request_body_bytes": len(json.dumps(body)),
                "prompt_chars": len(prompt_text),
                "tool_uses": tool_uses,
                "tool_results_chars": tool_results_chars,
            }

        try:
            return await _try_call("attempt_ok")
        except Exception as e1:
            self.log.warning("attempt_retry_once", extra={"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e1)})
            if self.cfg.retry_delay_s > 0:
                await asyncio.sleep(self.cfg.retry_delay_s)
            try:
                return await _try_call("attempt_ok_after_retry")
            except Exception as e2:
                self.log.error("attempt_failed", extra={"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e2)})
                return {"fn": fn, "length": L, "attempt": attempt_idx, "error": str(e2)}

    async def _call_once(self, fn: str, L: int, attempt_idx: int, data_examples: List[str], decimal: bool, tabular: bool = False) -> Dict[str, Any]:
        prompt_text = build_user_prompt(
            data_examples=data_examples,
            seq_len=L,
            decimal=decimal,
            tabular=tabular,
            prompt_variant=self.cfg.prompt_variant,
        )
        return await self._call_with_prompt(fn, L, attempt_idx, prompt_text)

    def _evaluate_candidate_code(
        self,
        code_str: Optional[str],
        seen_train_lines: List[str],
        current_batch_lines: List[str],
        val_lines: List[str],
        test_lines: List[str],
        is_tabular: bool,
    ) -> Dict[str, Any]:
        code_info = {
            "code": None,
            "code_lines": None,
            "num_branches": None,
            "code_analysis_error": None,
        }
        val_acc = None
        test_acc = None
        train_acc = None
        current_batch_train_acc = None
        compile_error = None

        if not code_str:
            compile_error = "no_code_found"
        else:
            code_info = analyze_code_structure(code_str, sanitize=self.cfg.sanitize_generated_code)
            try:
                fn_callable = compile_callable_from_code(code_str, sanitize=self.cfg.sanitize_generated_code)
                current_batch_train_acc = evaluate_accuracy(fn_callable, current_batch_lines, self.log, is_tabular)
                train_acc = evaluate_accuracy(fn_callable, seen_train_lines, self.log, is_tabular)
                val_acc = evaluate_accuracy(fn_callable, val_lines, self.log, is_tabular)
                test_acc = evaluate_accuracy(fn_callable, test_lines, self.log, is_tabular)
            except Exception as e:
                compile_error = str(e)

        return {
            **code_info,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "current_batch_train_acc": current_batch_train_acc,
            "compile_error": compile_error,
        }

    def _build_candidate_row(
        self,
        *,
        result: Dict[str, Any],
        trial: int,
        dataset_seed: int,
        batch_index: int,
        batch_count: int,
        batch_size_current: int,
        seen_train_size: int,
        candidate_source: str,
        stopped_early: bool,
        eval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = self._attach_meta(
            {
                **result,
                "trial": trial,
                "stopped_early": stopped_early,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "batch_size_current": batch_size_current,
                "seen_train_size": seen_train_size,
                "candidate_source": candidate_source,
                "selected_for_next_batch": False,
                "is_final_selected_model": False,
                **eval_result,
            },
            dataset_seed=dataset_seed,
        )
        return row

    async def _run_normal_trial(
        self,
        *,
        fn: str,
        L: int,
        trial: int,
        dataset_seed: int,
        train_lines: List[str],
        val_lines: List[str],
        test_lines: List[str],
        is_decimal: bool,
        is_tabular: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        trial_rows: List[Dict[str, Any]] = []
        best_val_acc = None
        best_row = None
        stopped_early = False

        for k in range(1, self.cfg.attempts + 1):
            res = await self._call_once(fn, L, k, train_lines, is_decimal, is_tabular)
            out_text = res.get("text") or ""
            code_str = extract_code_from_output(out_text)
            eval_result = self._evaluate_candidate_code(
                code_str=code_str,
                seen_train_lines=train_lines,
                current_batch_lines=train_lines,
                val_lines=val_lines,
                test_lines=test_lines,
                is_tabular=is_tabular,
            )
            if eval_result["compile_error"]:
                self.log.warning(
                    "compile_or_eval_error",
                    extra={"fn": fn, "length": L, "attempt": k, "trial": trial, "error": eval_result["compile_error"]},
                )
            row = self._build_candidate_row(
                result=res,
                trial=trial,
                dataset_seed=dataset_seed,
                batch_index=1,
                batch_count=1,
                batch_size_current=len(train_lines),
                seen_train_size=len(train_lines),
                candidate_source="generated",
                stopped_early=stopped_early,
                eval_result=eval_result,
            )
            trial_rows.append(row)

            if should_update_best_by_val(row.get("val_acc"), best_val_acc):
                best_val_acc = row.get("val_acc")
                best_row = row

            if row.get("val_acc") == 1.0:
                row["stopped_early"] = True
                stopped_early = True
                self.log.info(
                    "early_stop",
                    extra={"fn": fn, "length": L, "attempt": k, "trial": trial, "val_acc": row.get("val_acc"), "test_acc": row.get("test_acc")},
                )
                break

        if best_row is not None:
            best_row["is_final_selected_model"] = True
        return trial_rows, best_row, []

    async def _run_batched_trial(
        self,
        *,
        fn: str,
        L: int,
        trial: int,
        dataset_seed: int,
        train_lines: List[str],
        val_lines: List[str],
        test_lines: List[str],
        is_decimal: bool,
        is_tabular: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        batch_size = self.cfg.code0_batch_size
        batches = split_train_batches(train_lines, batch_size)
        trial_rows: List[Dict[str, Any]] = []
        batch_winner_rows: List[Dict[str, Any]] = []
        seen_train_lines: List[str] = []
        incumbent_code: Optional[str] = None
        incumbent_eval: Optional[Dict[str, Any]] = None
        attempt_counter = 0
        final_best_row: Optional[Dict[str, Any]] = None

        for batch_index, batch_lines in enumerate(batches, start=1):
            seen_train_lines.extend(batch_lines)
            self.log.info(
                "batch_start",
                extra={
                    "fn": fn,
                    "length": L,
                    "trial": trial,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "batch_size_current": len(batch_lines),
                    "seen_train_size": len(seen_train_lines),
                },
            )

            best_row = None
            best_train_acc = None
            best_batch_train_acc = None
            best_val_acc = None

            if incumbent_code is not None:
                incumbent_eval = self._evaluate_candidate_code(
                    code_str=incumbent_code,
                    seen_train_lines=seen_train_lines,
                    current_batch_lines=batch_lines,
                    val_lines=val_lines,
                    test_lines=test_lines,
                    is_tabular=is_tabular,
                )
                incumbent_row = self._build_candidate_row(
                    result={
                        "fn": fn,
                        "length": L,
                        "attempt": None,
                        "prompt": None,
                        "text": None,
                        "usage": {},
                        "cached_tokens": None,
                        "duration_ms": None,
                        "request_body_bytes": None,
                        "prompt_chars": None,
                        "tool_uses": 0,
                        "tool_results_chars": 0,
                    },
                    trial=trial,
                    dataset_seed=dataset_seed,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_size_current=len(batch_lines),
                    seen_train_size=len(seen_train_lines),
                    candidate_source="incumbent",
                    stopped_early=False,
                    eval_result=incumbent_eval,
                )
                trial_rows.append(incumbent_row)
                if should_update_best_by_train(
                    incumbent_row.get("train_acc"),
                    best_train_acc,
                    incumbent_row.get("current_batch_train_acc"),
                    best_batch_train_acc,
                    incumbent_row.get("val_acc"),
                    best_val_acc,
                ):
                    best_row = incumbent_row
                    best_train_acc = incumbent_row.get("train_acc")
                    best_batch_train_acc = incumbent_row.get("current_batch_train_acc")
                    best_val_acc = incumbent_row.get("val_acc")

            # Compute misclassified examples from incumbent on all seen data
            incumbent_misclassified: List[str] = []
            incumbent_train_acc: Optional[float] = None
            if incumbent_code is not None and batch_index > 1:
                incumbent_misclassified = find_misclassified_examples(
                    incumbent_code, seen_train_lines, is_tabular=is_tabular,
                    max_examples=10, sanitize=self.cfg.sanitize_generated_code,
                )
                incumbent_train_acc = incumbent_eval.get("train_acc") if incumbent_eval else None

            for _ in range(self.cfg.attempts):
                attempt_counter += 1
                if batch_index == 1:
                    res = await self._call_once(fn, L, attempt_counter, batch_lines, is_decimal, is_tabular)
                else:
                    prompt_text = build_batch_revision_prompt(
                        previous_code=incumbent_code or "",
                        data_examples=batch_lines,
                        seq_len=L,
                        decimal=is_decimal,
                        tabular=is_tabular,
                        prompt_variant=self.cfg.prompt_variant,
                        train_acc=incumbent_train_acc,
                        misclassified_examples=incumbent_misclassified,
                    )
                    res = await self._call_with_prompt(fn, L, attempt_counter, prompt_text)
                code_str = extract_code_from_output(res.get("text") or "")
                eval_result = self._evaluate_candidate_code(
                    code_str=code_str,
                    seen_train_lines=seen_train_lines,
                    current_batch_lines=batch_lines,
                    val_lines=val_lines,
                    test_lines=test_lines,
                    is_tabular=is_tabular,
                )
                if eval_result["compile_error"]:
                    self.log.warning(
                        "compile_or_eval_error",
                        extra={"fn": fn, "length": L, "attempt": attempt_counter, "trial": trial, "error": eval_result["compile_error"]},
                    )
                row = self._build_candidate_row(
                    result=res,
                    trial=trial,
                    dataset_seed=dataset_seed,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    batch_size_current=len(batch_lines),
                    seen_train_size=len(seen_train_lines),
                    candidate_source="generated",
                    stopped_early=False,
                    eval_result=eval_result,
                )
                trial_rows.append(row)

                if should_update_best_by_train(
                    row.get("train_acc"),
                    best_train_acc,
                    row.get("current_batch_train_acc"),
                    best_batch_train_acc,
                    row.get("val_acc"),
                    best_val_acc,
                ):
                    best_row = row
                    best_train_acc = row.get("train_acc")
                    best_batch_train_acc = row.get("current_batch_train_acc")
                    best_val_acc = row.get("val_acc")

            if best_row is None:
                self.log.warning(
                    "batch_no_winner",
                    extra={"fn": fn, "length": L, "trial": trial, "batch_index": batch_index},
                )
                break

            if batch_index < len(batches):
                best_row["selected_for_next_batch"] = True
            else:
                best_row["is_final_selected_model"] = True
            batch_winner_rows.append(best_row)
            incumbent_code = best_row.get("code")
            final_best_row = best_row
            self.log.info(
                "batch_winner",
                extra={
                    "fn": fn,
                    "length": L,
                    "trial": trial,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "candidate_source": best_row.get("candidate_source"),
                    "train_acc": best_row.get("train_acc"),
                    "current_batch_train_acc": best_row.get("current_batch_train_acc"),
                    "val_acc": best_row.get("val_acc"),
                    "test_acc": best_row.get("test_acc"),
                },
            )

        return trial_rows, final_best_row, batch_winner_rows

    async def run(self) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        
        base_jsonl_path = self.cfg.out_jsonl
        base_name, ext = os.path.splitext(base_jsonl_path)
        os.makedirs(os.path.dirname(base_jsonl_path) if os.path.dirname(base_jsonl_path) else ".", exist_ok=True)
        
        final_jsonl_file = open(base_jsonl_path, "w", encoding="utf-8")

        try:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    self.log.error("unknown_function", extra={"fn": fn})
                    continue
                
                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = task_meta.get("lengths", self.cfg.lengths)
                for L in current_lengths:
                    dataset_seed = self.ds.derived_seed(fn, L)
                    train_lines, val_lines, test_lines, is_decimal, is_tabular = self.ds.get(fn, L)

                    trial_test_accuracies = []
                    trial_val_accuracies = []
                    trial_train_accuracies = []
                    trial_best_rows = []

                    for trial in range(self.cfg.num_trials):
                        safe_fn = re.sub(r"[^A-Za-z0-9_]+", "_", fn)
                        trial_jsonl_path = f"{base_name}_{safe_fn}_L{L}_trial{trial+1}{ext}"
                        if self.cfg.code0_train_mode == "batched":
                            trial_rows, best_row, batch_rows = await self._run_batched_trial(
                                fn=fn,
                                L=L,
                                trial=trial + 1,
                                dataset_seed=dataset_seed,
                                train_lines=train_lines,
                                val_lines=val_lines,
                                test_lines=test_lines,
                                is_decimal=is_decimal,
                                is_tabular=is_tabular,
                            )
                            self.batch_winner_rows.extend(batch_rows)
                        else:
                            trial_rows, best_row, _ = await self._run_normal_trial(
                                fn=fn,
                                L=L,
                                trial=trial + 1,
                                dataset_seed=dataset_seed,
                                train_lines=train_lines,
                                val_lines=val_lines,
                                test_lines=test_lines,
                                is_decimal=is_decimal,
                                is_tabular=is_tabular,
                            )

                        with open(trial_jsonl_path, "w", encoding="utf-8") as trial_jsonl_file:
                            for row in trial_rows:
                                trial_jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        all_rows.extend(trial_rows)

                        if best_row and best_row.get("test_acc") is not None:
                            trial_test_accuracies.append(float(best_row["test_acc"]))
                            trial_val_accuracies.append(float(best_row["val_acc"]))
                            if best_row.get("train_acc") is not None:
                                trial_train_accuracies.append(float(best_row["train_acc"]))
                            trial_best_rows.append(best_row)

                    for best_row in trial_best_rows:
                        final_jsonl_file.write(json.dumps(best_row, ensure_ascii=False) + "\n")
                        final_jsonl_file.flush()

                    if trial_test_accuracies:
                        import numpy as np
                        test_acc_mean = float(np.mean(trial_test_accuracies))
                        test_acc_std = float(np.std(trial_test_accuracies))
                        val_acc_mean = float(np.mean(trial_val_accuracies))
                        val_acc_std = float(np.std(trial_val_accuracies))
                        train_acc_mean = float(np.mean(trial_train_accuracies)) if trial_train_accuracies else None
                        train_acc_std = float(np.std(trial_train_accuracies)) if trial_train_accuracies else None

                        summary_row = self._attach_meta({
                            "fn": fn,
                            "length": L,
                            "attempt": None,
                            "trial": None,
                            "prompt": None,
                            "text": None,
                            "code": None,
                            "code_lines": None,
                            "num_branches": None,
                            "code_analysis_error": None,
                            "duration_ms": None,
                            "cached_tokens": None,
                            "usage": {},
                            "tool_uses": None,
                            "tool_results_chars": None,
                            "val_acc": val_acc_mean,
                            "val_acc_std": val_acc_std,
                            "test_acc": test_acc_mean,
                            "test_acc_std": test_acc_std,
                            "train_acc": train_acc_mean,
                            "train_acc_std": train_acc_std,
                            "stopped_early": None,
                            "compile_error": None,
                            "num_trials": self.cfg.num_trials,
                            "is_summary": True,
                        }, dataset_seed=dataset_seed)

                        all_rows.append(summary_row)
                        final_jsonl_file.write(json.dumps(summary_row, ensure_ascii=False) + "\n")
                        final_jsonl_file.flush()

                        self.log.info(
                            f"{fn} L={L}: test_acc={test_acc_mean:.4f}+/-{test_acc_std:.4f} "
                            f"(mean+/-std over {self.cfg.num_trials} trials)"
                        )

            self.log.info("dispatch_finished", extra={"total_results": len(all_rows)})
        finally:
            final_jsonl_file.close()
        
        return all_rows


# =========================
# Writers
# =========================

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_id", "api_mode", "api_base_url", "azure_endpoint", "api_version", "client_type", "model", "returned_model", "reasoning_effort", "max_output_tokens", "tool_choice", "prompt_variant",
        "enable_code_interpreter", "global_seed", "dataset_seed", "train_size", "val_size", "test_size",
        "dataset_dir", "attempts_requested", "num_trials_requested", "dry_run", "code0_train_mode", "code0_batch_size",
        "fn", "length", "attempt", "trial", "prompt", "text",
        "code", "code_lines", "num_branches", "code_analysis_error",
        "duration_ms", "cached_tokens", "prompt_tokens", "completion_tokens",
        "reasoning_tokens", "tool_uses", "tool_results_chars",
        "pricing_available", "pricing_source", "pricing_model", "input_rate_per_million", "output_rate_per_million",
        "estimated_input_cost_usd", "estimated_output_cost_usd", "estimated_total_cost_usd",
        "batch_index", "batch_count", "batch_size_current", "seen_train_size", "current_batch_train_acc",
        "candidate_source", "selected_for_next_batch", "is_final_selected_model",
        "val_acc", "val_acc_std", "test_acc", "test_acc_std", "train_acc", "train_acc_std", "stopped_early", "compile_error", "num_trials", "is_summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            usage = r.get("usage") or {}
            cost_fields = shared_flatten_cost_fields(r.get("cost") or {})
            w.writerow({
                "run_id": r.get("run_id"),
                "api_mode": r.get("api_mode"),
                "api_base_url": r.get("api_base_url"),
                "azure_endpoint": r.get("azure_endpoint"),
                "api_version": r.get("api_version"),
                "client_type": r.get("client_type"),
                "model": r.get("model"),
                "returned_model": r.get("returned_model"),
                "reasoning_effort": r.get("reasoning_effort"),
                "max_output_tokens": r.get("max_output_tokens"),
                "tool_choice": r.get("tool_choice"),
                "prompt_variant": r.get("prompt_variant"),
                "enable_code_interpreter": r.get("enable_code_interpreter"),
                "global_seed": r.get("global_seed"),
                "dataset_seed": r.get("dataset_seed"),
                "train_size": r.get("train_size"),
                "val_size": r.get("val_size"),
                "test_size": r.get("test_size"),
                "dataset_dir": r.get("dataset_dir"),
                "attempts_requested": r.get("attempts_requested"),
                "num_trials_requested": r.get("num_trials_requested"),
                "dry_run": r.get("dry_run"),
                "code0_train_mode": r.get("code0_train_mode"),
                "code0_batch_size": r.get("code0_batch_size"),
                "fn": r.get("fn"),
                "length": r.get("length"),
                "attempt": r.get("attempt"),
                "trial": r.get("trial"),
                "prompt": r.get("prompt"),
                "text": r.get("text"),
                "code": r.get("code"),
                "code_lines": r.get("code_lines"),
                "num_branches": r.get("num_branches"),
                "code_analysis_error": r.get("code_analysis_error"),
                "duration_ms": r.get("duration_ms"),
                "cached_tokens": r.get("cached_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "tool_uses": r.get("tool_uses"),
                "tool_results_chars": r.get("tool_results_chars"),
                "pricing_available": cost_fields.get("pricing_available"),
                "pricing_source": cost_fields.get("pricing_source"),
                "pricing_model": cost_fields.get("pricing_model"),
                "input_rate_per_million": cost_fields.get("input_rate_per_million"),
                "output_rate_per_million": cost_fields.get("output_rate_per_million"),
                "estimated_input_cost_usd": cost_fields.get("estimated_input_cost_usd"),
                "estimated_output_cost_usd": cost_fields.get("estimated_output_cost_usd"),
                "estimated_total_cost_usd": cost_fields.get("estimated_total_cost_usd"),
                "batch_index": r.get("batch_index"),
                "batch_count": r.get("batch_count"),
                "batch_size_current": r.get("batch_size_current"),
                "seen_train_size": r.get("seen_train_size"),
                "current_batch_train_acc": r.get("current_batch_train_acc"),
                "candidate_source": r.get("candidate_source"),
                "selected_for_next_batch": r.get("selected_for_next_batch"),
                "is_final_selected_model": r.get("is_final_selected_model"),
                "val_acc": r.get("val_acc"),
                "val_acc_std": r.get("val_acc_std"),
                "test_acc": r.get("test_acc"),
                "test_acc_std": r.get("test_acc_std"),
                "train_acc": r.get("train_acc"),
                "train_acc_std": r.get("train_acc_std"),
                "stopped_early": r.get("stopped_early"),
                "compile_error": r.get("compile_error"),
                "num_trials": r.get("num_trials"),
                "is_summary": r.get("is_summary"),
            })

def write_manifest(path: str, cfg: Config, rows: List[Dict[str, Any]]) -> None:
    manifest = {
        "run_id": cfg.run_id,
        "created_ts": int(time.time()),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "api_key_set": bool(cfg.api_key),
        "openai_api_key_set": bool(cfg.api_key),
        "config": {
            "api_mode": cfg.api_mode,
            "api_base_url": cfg.api_base_url,
            "azure_endpoint": cfg.azure_endpoint,
            "api_version": cfg.api_version,
            "model": cfg.model,
            "max_output_tokens": cfg.max_output_tokens,
            "reasoning_effort": cfg.reasoning_effort,
            "verbosity": cfg.verbosity,
            "tool_choice": cfg.tool_choice,
            "prompt_variant": cfg.prompt_variant,
            "sanitize_generated_code": cfg.sanitize_generated_code,
            "enable_code_interpreter": cfg.enable_code_interpreter,
            "dry_run": cfg.dry_run,
            "concurrency": cfg.concurrency,
            "per_call_timeout_s": cfg.per_call_timeout_s,
            "retry_delay_s": cfg.retry_delay_s,
            "functions": cfg.functions,
            "lengths": cfg.lengths,
            "attempts": cfg.attempts,
            "num_trials": cfg.num_trials,
            "train_size": cfg.train_size,
            "val_size": cfg.val_size,
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "dataset_dir": cfg.dataset_dir,
            "out_jsonl": cfg.out_jsonl,
            "out_csv": cfg.out_csv,
            "code0_train_mode": cfg.code0_train_mode,
            "code0_batch_size": cfg.code0_batch_size if cfg.code0_train_mode == "batched" else None,
        },
        "artifacts": {
            "jsonl": cfg.out_jsonl,
            "csv": cfg.out_csv,
            "batch_jsonl": derive_batch_artifact_paths(cfg)[0],
            "batch_csv": derive_batch_artifact_paths(cfg)[1],
        },
        "row_stats": {
            "total_rows": len(rows),
            "summary_rows": sum(1 for r in rows if r.get("is_summary")),
            "attempt_rows": sum(1 for r in rows if r.get("attempt") is not None and not r.get("is_summary")),
            "compile_error_rows": sum(1 for r in rows if r.get("compile_error")),
            "final_selected_rows": sum(1 for r in rows if r.get("is_final_selected_model")),
            "batch_winner_rows": sum(1 for r in rows if r.get("selected_for_next_batch") or r.get("is_final_selected_model")),
            "attempt_estimated_total_cost_usd": sum(
                float(((r.get("cost") or {}).get("estimated_total_cost_usd")) or 0.0)
                for r in rows
                if r.get("attempt") is not None and not r.get("is_summary")
            ),
        },
    }
    _safe_write_json(path, manifest)


# =========================
# CLI
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="OpenAI runner (early-stop + persistent datasets)")

    p.add_argument("--api-mode", choices=["responses", "chat_completions"], help="API mode: Responses API or chat completions")
    p.add_argument("--api-base-url", help="Override API base URL (e.g., TAMU gateway base URL)")
    p.add_argument("--azure-endpoint", help="Azure OpenAI endpoint (e.g., https://...openai.azure.com/)")
    p.add_argument("--api-version", help="Azure OpenAI API version (default: 2024-12-01-preview)")
    p.add_argument("--functions", nargs="*", help="Function IDs (e.g., fn_a fn_b ...)")
    p.add_argument("--lengths", nargs="*", type=int, help="Sequence lengths (e.g., 100 50 30 25 20)")
    p.add_argument("--attempts", type=int, help="Attempts per (fn, length), default=5")
    p.add_argument("--num-trials", type=int, help="Number of trials per (fn, length) for statistics, default=5")
    p.add_argument("--concurrency", type=int, help="Max concurrent API calls (default: 5)")
    p.add_argument("--timeout", type=float, help="Per-call timeout seconds (default: 1200)")
    p.add_argument("--retry-delay", type=float, help="Seconds to wait before retrying a failed call (default: 5)")

    p.add_argument("--model", help="Model or Azure deployment name")
    p.add_argument("--max-output-tokens", type=int, help="Max output tokens (default: 20000)")
    p.add_argument("--enable-code-interpreter", action="store_true", help="Enable Code Interpreter tool")
    p.add_argument("--tool-choice", choices=["auto","none"], help="Tool choice (default: auto)")
    p.add_argument("--sanitize-generated-code", choices=["on", "off"], help="Normalize generated code before compile/analyze (default: on)")
    p.add_argument("--prompt-variant", choices=["standard", "explain", "interview", "preview", "multipath", "subgroups", "thesis_aware", "regional", "ensemble"], help="Prompt variant (default: standard)")
    p.add_argument("--verbosity", choices=["low","medium","high"], help="text.verbosity (default: low)")
    p.add_argument("--reasoning-effort", choices=["none","minimal","low","medium","high","xhigh"], help="reasoning.effort (default: high)")

    p.add_argument("--train-size", type=int, help="Train size per (fn, L) (default: 100)")
    p.add_argument("--val-size", type=int, help="Validation size per (fn, L) (default: 100)")
    p.add_argument("--test-size", type=int, help="Test size per (fn, L) (default: 10000)")
    p.add_argument("--code0-train-mode", choices=["normal", "batched"], help="Code0 training mode (default: normal)")
    p.add_argument("--code0-batch-size", type=int, help="Train batch size for batched Code0 mode")
    p.add_argument("--seed", type=int, help="Global seed (default: 42)")
    p.add_argument("--dataset-dir", type=str, help="Dataset root directory (default: program_synthesis/datasets)")

    p.add_argument("--out-jsonl", help="Output JSONL path (default: program_synthesis/results_attempts.jsonl)")
    p.add_argument("--out-csv", help="Output CSV path (default: program_synthesis/results_attempts.csv)")
    p.add_argument("--out-manifest", help="Output manifest JSON path (default: <out-jsonl>_manifest.json)")
    p.add_argument("--run-id", help="Optional run id for artifact traceability")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level (default: INFO)")
    p.add_argument("--dry-run", action="store_true", help="Dry run, shows input prompt generated for each query")
    p.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode (adds chat_template_kwargs for Qwen/DeepSeek models)")

    args = p.parse_args()
    cfg = Config()

    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.api_mode: cfg.api_mode = args.api_mode
    if args.api_base_url: cfg.api_base_url = args.api_base_url
    if args.azure_endpoint: cfg.azure_endpoint = args.azure_endpoint.strip()
    if args.api_version: cfg.api_version = args.api_version.strip()
    if args.attempts: cfg.attempts = args.attempts
    if args.num_trials: cfg.num_trials = args.num_trials
    if args.concurrency: cfg.concurrency = args.concurrency
    if args.model: cfg.model = args.model
    if args.max_output_tokens: cfg.max_output_tokens = args.max_output_tokens
    if args.enable_code_interpreter: cfg.enable_code_interpreter = True
    if args.tool_choice: cfg.tool_choice = args.tool_choice
    if args.sanitize_generated_code is not None:
        cfg.sanitize_generated_code = args.sanitize_generated_code == "on"
    if args.prompt_variant: cfg.prompt_variant = args.prompt_variant
    if args.out_jsonl: cfg.out_jsonl = args.out_jsonl
    if args.out_csv: cfg.out_csv = args.out_csv
    if args.out_manifest: cfg.out_manifest = args.out_manifest
    if args.run_id: cfg.run_id = args.run_id
    if args.verbosity: cfg.verbosity = args.verbosity
    if args.reasoning_effort: cfg.reasoning_effort = args.reasoning_effort
    if args.timeout: cfg.per_call_timeout_s = args.timeout
    if args.retry_delay is not None: cfg.retry_delay_s = args.retry_delay
    if args.train_size: cfg.train_size = args.train_size
    if args.val_size: cfg.val_size = args.val_size
    if args.test_size: cfg.test_size = args.test_size
    if args.code0_train_mode: cfg.code0_train_mode = args.code0_train_mode
    if args.code0_batch_size is not None: cfg.code0_batch_size = args.code0_batch_size
    if args.seed is not None: cfg.seed = args.seed
    if args.dataset_dir: cfg.dataset_dir = args.dataset_dir
    if args.dry_run: cfg.dry_run = True
    if args.enable_thinking: cfg.enable_thinking = True
    if cfg.code0_train_mode == "batched" and cfg.code0_batch_size <= 0:
        p.error("--code0-batch-size must be positive when --code0-train-mode batched is used")

    os.environ["LOG_LEVEL"] = args.log_level
    return cfg


async def _amain(cfg: Config) -> None:
    log = setup_logger(os.getenv("LOG_LEVEL", "INFO"))
    runner = Runner(cfg, log)
    try:
        rows = await runner.run()
    finally:
        await runner.client.close()
    write_csv(cfg.out_csv, rows)
    batch_jsonl_path, batch_csv_path = derive_batch_artifact_paths(cfg)
    write_jsonl(batch_jsonl_path, runner.batch_winner_rows)
    write_csv(batch_csv_path, runner.batch_winner_rows)
    manifest_path = cfg.out_manifest or (os.path.splitext(cfg.out_jsonl)[0] + "_manifest.json")
    write_manifest(manifest_path, cfg, rows)
    log.info(
        "artifacts_written",
        extra={
            "jsonl": cfg.out_jsonl,
            "csv": cfg.out_csv,
            "batch_jsonl": batch_jsonl_path,
            "batch_csv": batch_csv_path,
            "manifest": manifest_path,
            "run_id": cfg.run_id,
        },
    )


def main() -> None:
    cfg = parse_args()
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
