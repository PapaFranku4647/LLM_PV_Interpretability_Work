from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import httpx


_PRICING_SOURCE = "tamus_ai_chat_docs_2026-03-19"
_DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
_DEFAULT_TAMU_DEPLOYMENT = "gpt-5.2-deep-learning-fundamentals"
_NUMERIC_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
)
_DETAIL_KEYS = (
    "prompt_tokens_details",
    "input_token_details",
    "output_tokens_details",
    "completion_tokens_details",
)

_MODEL_PRICING_TABLE: list[tuple[tuple[str, ...], str, float, float]] = [
    (("claude-3.5-haiku", "claude35haiku"), "claude-3.5-haiku", 0.80, 4.00),
    (("claude-3.5-sonnet", "claude35sonnet"), "claude-3.5-sonnet", 3.00, 15.00),
    (("claude-3.7-sonnet", "claude37sonnet"), "claude-3.7-sonnet", 3.00, 15.00),
    (("claude-haiku-4.5", "claude-4.5-haiku", "claudehaiku45"), "claude-haiku-4.5", 1.00, 5.00),
    (("claude-sonnet-4.5", "claude-4.5-sonnet", "claudesonnet45"), "claude-sonnet-4.5", 3.00, 15.00),
    (("claude-sonnet-4", "claude-4-sonnet", "claudesonnet4"), "claude-sonnet-4", 3.00, 15.00),
    (("claude-opus-4.5", "claude-4.5-opus", "claudeopus45"), "claude-opus-4.5", 5.00, 25.00),
    (("claude-opus-4.1", "claude-4.1-opus", "claudeopus41"), "claude-opus-4.1", 15.00, 75.00),
    (("gemini-2.0-flash-lite", "gemini20flashlite"), "gemini-2.0-flash-lite", 0.075, 0.30),
    (("gemini-2.5-flash-lite", "gemini25flashlite"), "gemini-2.5-flash-lite", 0.10, 0.40),
    (("gemini-2.0-flash", "gemini20flash"), "gemini-2.0-flash", 0.15, 0.60),
    (("gemini-2.5-flash", "gemini25flash"), "gemini-2.5-flash", 0.15, 0.60),
    (("gemini-2.5-pro", "gemini25pro"), "gemini-2.5-pro", 1.25, 10.00),
    (("gpt-4o", "gpt4o"), "gpt-4o", 2.50, 10.00),
    (("gpt-4.1", "gpt41"), "gpt-4.1", 2.00, 8.00),
    (("gpt-5.2", "gpt52"), "gpt-5.2", 1.75, 14.00),
    (("gpt-5.1", "gpt51"), "gpt-5.1", 1.25, 10.00),
    (("gpt-5", "gpt5"), "gpt-5", 1.25, 10.00),
    (("o3-mini", "o3mini"), "o3-mini", 1.10, 4.40),
    (("o4-mini", "o4mini"), "o4-mini", 1.10, 4.40),
    (("llama-3.2", "llama3.2", "llama32"), "llama-3.2", 1.00, 3.00),
]


@dataclass(frozen=True)
class ModelPricing:
    canonical_name: str
    input_per_million: float
    output_per_million: float
    source: str = _PRICING_SOURCE


@dataclass
class LLMCallResult:
    text: str
    usage: dict[str, Any]
    response_dump: dict[str, Any]
    response_id: Optional[str] = None
    response_status: Optional[str] = None
    model: Optional[str] = None
    tool_uses: int = 0
    tool_results_chars: int = 0


def endpoint_is_azure(endpoint: str | None) -> bool:
    return ".openai.azure.com" in ((endpoint or "").strip().lower())


def resolve_azure_endpoint_from_env() -> str:
    return (
        os.getenv("TAMU_AZURE_ENDPOINT")
        or os.getenv("DPF_URL")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()


def resolve_api_version_from_env() -> str:
    return (os.getenv("TAMU_API_VERSION") or _DEFAULT_AZURE_API_VERSION).strip()


def resolve_default_model_from_env(default_non_azure: str) -> str:
    explicit = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("TAMU_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    if explicit:
        return explicit.strip()
    if resolve_azure_endpoint_from_env() or endpoint_is_azure(os.getenv("API_BASE_URL", "")):
        return _DEFAULT_TAMU_DEPLOYMENT
    return default_non_azure


def infer_default_api_mode() -> str:
    explicit = os.getenv("API_MODE")
    if explicit:
        return explicit.strip().lower()
    if resolve_azure_endpoint_from_env() or endpoint_is_azure(os.getenv("API_BASE_URL", "")):
        return "chat_completions"
    return "responses"


def build_async_client(
    api_key: str,
    *,
    api_base_url: str = "",
    azure_endpoint: str = "",
    api_version: str | None = None,
) -> tuple[Any, str]:
    openai_mod = importlib.import_module("openai")
    resolved_azure_endpoint = azure_endpoint.strip() or (api_base_url.strip() if endpoint_is_azure(api_base_url) else "")
    if resolved_azure_endpoint:
        client_cls = getattr(openai_mod, "AsyncAzureOpenAI")
        return (
            client_cls(
                api_key=api_key,
                azure_endpoint=resolved_azure_endpoint,
                api_version=api_version or resolve_api_version_from_env(),
            ),
            "azure",
        )

    client_cls = getattr(openai_mod, "AsyncOpenAI")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base_url.strip():
        client_kwargs["base_url"] = api_base_url.strip()
    return client_cls(**client_kwargs), "openai"


def build_sync_client(
    api_key: str,
    *,
    api_base_url: str = "",
    azure_endpoint: str = "",
    api_version: str | None = None,
) -> tuple[Any, str]:
    openai_mod = importlib.import_module("openai")
    resolved_azure_endpoint = azure_endpoint.strip() or (api_base_url.strip() if endpoint_is_azure(api_base_url) else "")
    if resolved_azure_endpoint:
        client_cls = getattr(openai_mod, "AzureOpenAI")
        return (
            client_cls(
                api_key=api_key,
                azure_endpoint=resolved_azure_endpoint,
                api_version=api_version or resolve_api_version_from_env(),
            ),
            "azure",
        )

    client_cls = getattr(openai_mod, "OpenAI")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base_url.strip():
        client_kwargs["base_url"] = api_base_url.strip()
    return client_cls(**client_kwargs), "openai"


def build_chat_completions_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    reasoning_effort: str | None = None,
    stream: bool = False,
    response_format: Optional[dict[str, Any]] = None,
    enable_thinking: bool = False,
    api_base_url: str = "",
    azure_endpoint: str = "",
) -> dict[str, Any]:
    use_azure_shape = bool(azure_endpoint.strip()) or endpoint_is_azure(api_base_url)
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if use_azure_shape:
        body["max_completion_tokens"] = max_output_tokens
    else:
        body["max_tokens"] = max_output_tokens
    if reasoning_effort and chat_completion_supports_reasoning_effort(model):
        body["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        body["response_format"] = response_format
    if enable_thinking and not use_azure_shape:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body


def _normalize_model_name(model_name: str | None) -> str:
    lowered = (model_name or "").strip().lower()
    for prefix in ("protected.", "openai/", "anthropic/", "google/", "meta/"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
    return lowered.replace("_", "-").replace(" ", "")


def resolve_model_pricing(model_name: str | None) -> Optional[ModelPricing]:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return None
    for aliases, canonical_name, input_rate, output_rate in _MODEL_PRICING_TABLE:
        if any(alias in normalized for alias in aliases):
            return ModelPricing(
                canonical_name=canonical_name,
                input_per_million=input_rate,
                output_per_million=output_rate,
            )
    return None


def normalize_usage(usage_obj: Any) -> dict[str, Any]:
    if not usage_obj:
        return {}
    if isinstance(usage_obj, Mapping):
        usage = dict(usage_obj)
    else:
        usage = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            value = getattr(usage_obj, key, None)
            if value is not None:
                usage[key] = value

        for detail_key in _DETAIL_KEYS:
            detail = _coerce_detail_mapping(getattr(usage_obj, detail_key, None))
            if detail:
                usage[detail_key] = detail

    if "prompt_tokens" not in usage and "input_tokens" in usage:
        usage["prompt_tokens"] = usage["input_tokens"]
    if "completion_tokens" not in usage and "output_tokens" in usage:
        usage["completion_tokens"] = usage["output_tokens"]

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_token_details") or {}
    if isinstance(prompt_details, Mapping) and "cached_tokens" in prompt_details:
        usage["cached_tokens"] = prompt_details["cached_tokens"]

    if "reasoning_tokens" not in usage or usage.get("reasoning_tokens") is None:
        for detail_key in ("output_tokens_details", "completion_tokens_details", "input_token_details"):
            detail = usage.get(detail_key) or {}
            if isinstance(detail, Mapping) and detail.get("reasoning_tokens") is not None:
                usage["reasoning_tokens"] = detail["reasoning_tokens"]
                break

    return usage


def merge_usage(usages: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for raw_usage in usages:
        usage = normalize_usage(raw_usage or {})
        if not usage:
            continue
        for key in _NUMERIC_USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + int(value)
        for detail_key in _DETAIL_KEYS:
            detail = usage.get(detail_key)
            if not isinstance(detail, Mapping):
                continue
            merged_detail = merged.setdefault(detail_key, {})
            for sub_key, sub_value in detail.items():
                if isinstance(sub_value, (int, float)):
                    merged_detail[sub_key] = merged_detail.get(sub_key, 0) + int(sub_value)
    return normalize_usage(merged)


def estimate_usage_cost(
    usage: Mapping[str, Any] | None,
    model_name: str | None,
    *,
    input_rate: float | None = None,
    output_rate: float | None = None,
) -> dict[str, Any]:
    normalized_usage = normalize_usage(usage or {})
    pricing = resolve_model_pricing(model_name)
    resolved_input_rate = input_rate if input_rate is not None else (pricing.input_per_million if pricing else None)
    resolved_output_rate = output_rate if output_rate is not None else (pricing.output_per_million if pricing else None)

    prompt_tokens = int(normalized_usage.get("prompt_tokens") or 0)
    completion_tokens = int(normalized_usage.get("completion_tokens") or 0)
    input_cost = None
    output_cost = None
    total_cost = None
    if resolved_input_rate is not None:
        input_cost = (prompt_tokens / 1_000_000.0) * resolved_input_rate
    if resolved_output_rate is not None:
        output_cost = (completion_tokens / 1_000_000.0) * resolved_output_rate
    if input_cost is not None and output_cost is not None:
        total_cost = input_cost + output_cost

    return {
        "pricing_available": pricing is not None or (resolved_input_rate is not None and resolved_output_rate is not None),
        "pricing_source": pricing.source if pricing else ("manual_override" if resolved_input_rate is not None and resolved_output_rate is not None else None),
        "pricing_model": pricing.canonical_name if pricing else None,
        "input_rate_per_million": resolved_input_rate,
        "output_rate_per_million": resolved_output_rate,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": total_cost,
    }


def flatten_cost_fields(cost: Mapping[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    payload = dict(cost or {})
    return {
        f"{prefix}pricing_available": payload.get("pricing_available"),
        f"{prefix}pricing_source": payload.get("pricing_source"),
        f"{prefix}pricing_model": payload.get("pricing_model"),
        f"{prefix}input_rate_per_million": payload.get("input_rate_per_million"),
        f"{prefix}output_rate_per_million": payload.get("output_rate_per_million"),
        f"{prefix}estimated_input_cost_usd": payload.get("estimated_input_cost_usd"),
        f"{prefix}estimated_output_cost_usd": payload.get("estimated_output_cost_usd"),
        f"{prefix}estimated_total_cost_usd": payload.get("estimated_total_cost_usd"),
    }


def merge_cost_estimates(costs: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    valid_costs = [dict(cost or {}) for cost in costs if cost]
    if not valid_costs:
        return {}

    input_costs = [float(c["estimated_input_cost_usd"]) for c in valid_costs if c.get("estimated_input_cost_usd") is not None]
    output_costs = [float(c["estimated_output_cost_usd"]) for c in valid_costs if c.get("estimated_output_cost_usd") is not None]
    total_costs = [float(c["estimated_total_cost_usd"]) for c in valid_costs if c.get("estimated_total_cost_usd") is not None]
    pricing_models = sorted({str(c.get("pricing_model")) for c in valid_costs if c.get("pricing_model")})
    pricing_sources = sorted({str(c.get("pricing_source")) for c in valid_costs if c.get("pricing_source")})
    input_rates = sorted({float(c["input_rate_per_million"]) for c in valid_costs if c.get("input_rate_per_million") is not None})
    output_rates = sorted({float(c["output_rate_per_million"]) for c in valid_costs if c.get("output_rate_per_million") is not None})

    return {
        "pricing_available": bool(valid_costs) and all(bool(c.get("pricing_available")) for c in valid_costs),
        "pricing_source": pricing_sources[0] if len(pricing_sources) == 1 else ",".join(pricing_sources) if pricing_sources else None,
        "pricing_model": pricing_models[0] if len(pricing_models) == 1 else ",".join(pricing_models) if pricing_models else None,
        "input_rate_per_million": input_rates[0] if len(input_rates) == 1 else None,
        "output_rate_per_million": output_rates[0] if len(output_rates) == 1 else None,
        "estimated_input_cost_usd": sum(input_costs) if input_costs else None,
        "estimated_output_cost_usd": sum(output_costs) if output_costs else None,
        "estimated_total_cost_usd": sum(total_costs) if total_costs else None,
    }


def flatten_usage_fields(usage: Mapping[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    normalized_usage = normalize_usage(usage or {})
    return {
        f"{prefix}prompt_tokens": normalized_usage.get("prompt_tokens"),
        f"{prefix}completion_tokens": normalized_usage.get("completion_tokens"),
        f"{prefix}total_tokens": normalized_usage.get("total_tokens"),
        f"{prefix}reasoning_tokens": normalized_usage.get("reasoning_tokens"),
        f"{prefix}cached_tokens": normalized_usage.get("cached_tokens"),
    }


def chat_completion_supports_reasoning_effort(model_name: str) -> bool:
    model_l = (model_name or "").strip().lower()
    return (
        model_l.startswith("o3")
        or model_l.startswith("o4")
        or "gpt-5" in model_l
        or model_l.startswith("openai/o3")
        or model_l.startswith("openai/o4")
        or model_l.startswith("protected.o3")
        or model_l.startswith("protected.o4")
    )


def extract_text_from_response(response: Any) -> str:
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
                    text_value = _obj_get(chunk, "text")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
                    elif isinstance(text_value, dict):
                        nested = text_value.get("value")
                        if isinstance(nested, str) and nested.strip():
                            parts.append(nested.strip())
    return "\n".join(parts).strip()


def parse_chat_completion_sse(raw_text: str) -> dict[str, Any]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    model_name: Optional[str] = None
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if isinstance(payload.get("model"), str) and payload.get("model"):
            model_name = payload.get("model")
        chunk_usage = payload.get("usage")
        if isinstance(chunk_usage, Mapping):
            usage = dict(chunk_usage)
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                text_parts.extend(_extract_text_parts_from_choice(choice))
    return {
        "text": "".join(text_parts).strip(),
        "usage": usage,
        "model": model_name,
    }


def extract_text_from_chat_completion(result: Any) -> str:
    if isinstance(result, str):
        return parse_chat_completion_sse(result).get("text", "")
    choices = _obj_get(result, "choices") or []
    if not choices:
        return ""
    first_choice = choices[0]
    direct_parts = _extract_text_parts_from_choice(first_choice)
    if direct_parts:
        return "".join(direct_parts).strip()
    message = _obj_get(first_choice, "message")
    if message is None:
        return ""
    return "".join(_flatten_chat_content_text(_obj_get(message, "content"))).strip()


def parse_chat_completion_http_payload(payload_text: str, content_type: str = "") -> dict[str, Any]:
    text = payload_text or ""
    lowered_content_type = (content_type or "").lower()
    if "text/event-stream" in lowered_content_type or text.lstrip().startswith("data:"):
        parsed = parse_chat_completion_sse(text)
        return {
            "text": parsed.get("text", ""),
            "usage": normalize_usage(parsed.get("usage") or {}),
            "response_dump": {"raw_sse": text},
            "model": parsed.get("model"),
        }

    data = json.loads(text)
    usage = normalize_usage(data.get("usage") or {}) if isinstance(data, Mapping) else {}
    response_dump = dict(data) if isinstance(data, Mapping) else {"raw_payload": text}
    return {
        "text": extract_text_from_chat_completion(data),
        "usage": usage,
        "response_dump": response_dump,
        "model": response_dump.get("model"),
    }


async def call_llm_async(
    *,
    client: Any,
    api_mode: str,
    body: dict[str, Any],
    timeout: float,
    api_base_url: str = "",
    api_key: str = "",
    azure_endpoint: str = "",
) -> LLMCallResult:
    if api_mode == "chat_completions":
        if should_use_raw_chat_completions(api_base_url=api_base_url, azure_endpoint=azure_endpoint):
            return await _call_chat_completions_raw_async(
                api_base_url=api_base_url,
                api_key=api_key,
                body=body,
                timeout=timeout,
            )
        result = await client.chat.completions.create(**body)
        return _coerce_chat_completion_result(result)

    response = await client.responses.create(**body)
    return _coerce_responses_result(response)


def call_llm_sync(
    *,
    client: Any,
    api_mode: str,
    body: dict[str, Any],
    timeout: float,
    api_base_url: str = "",
    api_key: str = "",
    azure_endpoint: str = "",
) -> LLMCallResult:
    if api_mode == "chat_completions":
        if should_use_raw_chat_completions(api_base_url=api_base_url, azure_endpoint=azure_endpoint):
            return _call_chat_completions_raw_sync(
                api_base_url=api_base_url,
                api_key=api_key,
                body=body,
                timeout=timeout,
            )
        response = client.chat.completions.create(**body)
        return _coerce_chat_completion_result(response)

    response = client.responses.create(**body)
    return _coerce_responses_result(response)


def resolve_api_key_from_env() -> str:
    return (
        os.getenv("TAMU_API_KEY")
        or os.getenv("TAMUS_AI_CHAT_API_KEY")
        or os.getenv("dpf-key")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


def should_use_raw_chat_completions(
    *,
    api_base_url: str = "",
    azure_endpoint: str = "",
) -> bool:
    return bool(api_base_url.strip()) and not endpoint_is_azure(api_base_url) and not azure_endpoint.strip()


def _coerce_detail_mapping(detail_obj: Any) -> Optional[dict[str, Any]]:
    if detail_obj is None:
        return None
    if isinstance(detail_obj, Mapping):
        return dict(detail_obj)
    detail: dict[str, Any] = {}
    for key in ("cached_tokens", "audio_tokens", "reasoning_tokens"):
        value = getattr(detail_obj, key, None)
        if value is not None:
            detail[key] = value
    return detail or None


def _obj_get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _flatten_chat_content_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, Mapping):
        nested_text = content.get("text")
        if isinstance(nested_text, str):
            return [nested_text]
        return _flatten_chat_content_text(content.get("content"))
    if isinstance(content, list):
        parts: list[str] = []
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


def _extract_text_parts_from_choice(choice: Any) -> list[str]:
    parts: list[str] = []
    parts.extend(_flatten_chat_content_text(_obj_get(choice, "text")))

    delta = _obj_get(choice, "delta")
    if delta is not None:
        parts.extend(_flatten_chat_content_text(_obj_get(delta, "content")))

    message = _obj_get(choice, "message")
    if message is not None:
        parts.extend(_flatten_chat_content_text(_obj_get(message, "content")))

    return [part for part in parts if isinstance(part, str) and part]


def _coerce_chat_completion_result(result: Any) -> LLMCallResult:
    if hasattr(result, "__aiter__"):
        raise TypeError("Async iterators are not supported in the sync coercion path.")
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, bytearray, Mapping)) and not hasattr(result, "choices"):
        text_parts: list[str] = []
        usage_obj: dict[str, Any] = {}
        model_name: Optional[str] = None
        for chunk in result:
            payload = _coerce_mapping(chunk)
            if not payload:
                continue
            if isinstance(payload.get("model"), str) and payload.get("model"):
                model_name = payload.get("model")
            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text_parts.extend(_extract_text_parts_from_choice(choice))
            chunk_usage = payload.get("usage")
            if isinstance(chunk_usage, Mapping):
                usage_obj = dict(chunk_usage)
        return LLMCallResult(
            text="".join(text_parts).strip(),
            usage=normalize_usage(usage_obj),
            response_dump={"stream_chunks": None},
            model=model_name,
        )

    if isinstance(result, str):
        parsed = parse_chat_completion_sse(result)
        return LLMCallResult(
            text=parsed.get("text", ""),
            usage=normalize_usage(parsed.get("usage") or {}),
            response_dump={"raw_sse": result},
            model=parsed.get("model"),
        )

    if isinstance(result, (bytes, bytearray)):
        decoded = bytes(result).decode("utf-8", errors="ignore")
        parsed = parse_chat_completion_sse(decoded)
        return LLMCallResult(
            text=parsed.get("text", ""),
            usage=normalize_usage(parsed.get("usage") or {}),
            response_dump={"raw_sse": decoded},
            model=parsed.get("model"),
        )

    response_dump = _coerce_mapping(result) or {}
    usage = normalize_usage(_obj_get(result, "usage") or response_dump.get("usage") or {})
    response_id = _obj_get(result, "id") or response_dump.get("id")
    model_name = _obj_get(result, "model") or response_dump.get("model")
    return LLMCallResult(
        text=extract_text_from_chat_completion(result),
        usage=usage,
        response_dump=response_dump,
        response_id=response_id,
        response_status=_obj_get(result, "status") or response_dump.get("status"),
        model=model_name,
    )


def _coerce_responses_result(response: Any) -> LLMCallResult:
    response_dump = _coerce_mapping(response) or {}
    usage = normalize_usage(_obj_get(response, "usage") or response_dump.get("usage") or {})
    tool_uses, tool_results_chars = _count_response_tools(response)
    return LLMCallResult(
        text=extract_text_from_response(response),
        usage=usage,
        response_dump=response_dump,
        response_id=_obj_get(response, "id") or response_dump.get("id"),
        response_status=_obj_get(response, "status") or response_dump.get("status"),
        model=_obj_get(response, "model") or response_dump.get("model"),
        tool_uses=tool_uses,
        tool_results_chars=tool_results_chars,
    )


def _coerce_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _count_response_tools(response: Any) -> tuple[int, int]:
    tool_uses = 0
    tool_results_chars = 0
    output = _obj_get(response, "output") or []
    for item in output:
        item_type = _obj_get(item, "type")
        if item_type == "tool_use":
            tool_uses += 1
            continue
        if item_type != "tool_result":
            continue
        content = _obj_get(item, "content")
        if isinstance(content, list):
            for part in content:
                text = _obj_get(part, "text")
                if isinstance(text, str):
                    tool_results_chars += len(text)
        elif isinstance(content, str):
            tool_results_chars += len(content)
    return tool_uses, tool_results_chars


async def _call_chat_completions_raw_async(
    *,
    api_base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> LLMCallResult:
    url = api_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            parsed = parse_chat_completion_http_payload(
                response.text,
                response.headers.get("content-type", ""),
            )
            if parsed.get("text") or body.get("stream") is True:
                response_dump = parsed.get("response_dump") or {}
                return LLMCallResult(
                    text=parsed.get("text", ""),
                    usage=normalize_usage(parsed.get("usage") or {}),
                    response_dump=response_dump,
                    response_id=response_dump.get("id"),
                    response_status=response_dump.get("status"),
                    model=parsed.get("model") or response_dump.get("model"),
                )
            content_type = (response.headers.get("content-type", "") or "").lower()
            if "text/event-stream" not in content_type:
                response_dump = parsed.get("response_dump") or {}
                return LLMCallResult(
                    text=parsed.get("text", ""),
                    usage=normalize_usage(parsed.get("usage") or {}),
                    response_dump=response_dump,
                    response_id=response_dump.get("id"),
                    response_status=response_dump.get("status"),
                    model=parsed.get("model") or response_dump.get("model"),
                )
        except httpx.HTTPStatusError:
            if body.get("stream") is True:
                raise

        retry_body = dict(body)
        retry_body["stream"] = True
        retry_body["stream_options"] = {"include_usage": True}
        async with client.stream("POST", url, headers=headers, json=retry_body) as response:
            response.raise_for_status()
            chunks: list[str] = []
            async for chunk in response.aiter_text():
                chunks.append(chunk)
            parsed = parse_chat_completion_http_payload(
                "".join(chunks),
                response.headers.get("content-type", ""),
            )
            response_dump = parsed.get("response_dump") or {}
            return LLMCallResult(
                text=parsed.get("text", ""),
                usage=normalize_usage(parsed.get("usage") or {}),
                response_dump=response_dump,
                response_id=response_dump.get("id"),
                response_status=response_dump.get("status"),
                model=parsed.get("model") or response_dump.get("model"),
            )


def _call_chat_completions_raw_sync(
    *,
    api_base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> LLMCallResult:
    url = api_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        response.raise_for_status()
        parsed = parse_chat_completion_http_payload(
            response.text,
            response.headers.get("content-type", ""),
        )
        if parsed.get("text") or body.get("stream") is True:
            response_dump = parsed.get("response_dump") or {}
            return LLMCallResult(
                text=parsed.get("text", ""),
                usage=normalize_usage(parsed.get("usage") or {}),
                response_dump=response_dump,
                response_id=response_dump.get("id"),
                response_status=response_dump.get("status"),
                model=parsed.get("model") or response_dump.get("model"),
            )
        content_type = (response.headers.get("content-type", "") or "").lower()
        if "text/event-stream" not in content_type:
            response_dump = parsed.get("response_dump") or {}
            return LLMCallResult(
                text=parsed.get("text", ""),
                usage=normalize_usage(parsed.get("usage") or {}),
                response_dump=response_dump,
                response_id=response_dump.get("id"),
                response_status=response_dump.get("status"),
                model=parsed.get("model") or response_dump.get("model"),
            )
    except httpx.HTTPStatusError:
        if body.get("stream") is True:
            raise

    retry_body = dict(body)
    retry_body["stream"] = True
    retry_body["stream_options"] = {"include_usage": True}
    with httpx.stream("POST", url, headers=headers, json=retry_body, timeout=timeout) as response:
        response.raise_for_status()
        raw_chunks = list(response.iter_text())
        parsed = parse_chat_completion_http_payload(
            "".join(raw_chunks),
            response.headers.get("content-type", ""),
        )
        response_dump = parsed.get("response_dump") or {}
        return LLMCallResult(
            text=parsed.get("text", ""),
            usage=normalize_usage(parsed.get("usage") or {}),
            response_dump=response_dump,
            response_id=response_dump.get("id"),
            response_status=response_dump.get("status"),
            model=parsed.get("model") or response_dump.get("model"),
        )
