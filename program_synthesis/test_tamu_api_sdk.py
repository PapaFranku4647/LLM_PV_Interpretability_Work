import argparse
import httpx
import json
import os
import sys
from typing import Any, Mapping

from openai import APIError, APIStatusError, OpenAI


def flatten_chat_content_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, Mapping):
        nested = content.get("text")
        if isinstance(nested, str):
            return [nested]
        return flatten_chat_content_text(content.get("content"))
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.extend(flatten_chat_content_text(item))
        return parts
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return [text_attr]
    nested_attr = getattr(content, "content", None)
    if nested_attr is not None:
        return flatten_chat_content_text(nested_attr)
    return []


def extract_text_parts_from_choice(choice: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(choice, Mapping):
        delta = choice.get("delta")
        message = choice.get("message")
        text_value = choice.get("text")
    else:
        delta = getattr(choice, "delta", None)
        message = getattr(choice, "message", None)
        text_value = getattr(choice, "text", None)

    parts.extend(flatten_chat_content_text(text_value))

    if delta is not None:
        if isinstance(delta, Mapping):
            parts.extend(flatten_chat_content_text(delta.get("content")))
        else:
            parts.extend(flatten_chat_content_text(getattr(delta, "content", None)))

    if message is not None:
        if isinstance(message, Mapping):
            parts.extend(flatten_chat_content_text(message.get("content")))
        else:
            parts.extend(flatten_chat_content_text(getattr(message, "content", None)))

    return [p for p in parts if isinstance(p, str) and p]


def parse_chat_completion_sse(raw_text: str) -> dict[str, Any]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                text_parts.extend(extract_text_parts_from_choice(choice))
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, Mapping):
            usage = dict(chunk_usage)
    return {"text": "".join(text_parts).strip(), "usage": usage}


def extract_text(result: Any) -> tuple[str, Any]:
    if hasattr(result, "__iter__") and not hasattr(result, "choices"):
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        for chunk in result:
            if hasattr(chunk, "model_dump"):
                data = chunk.model_dump()
            elif isinstance(chunk, Mapping):
                data = dict(chunk)
            elif isinstance(chunk, str):
                data = parse_chat_completion_sse(chunk)
                return data.get("text", ""), data.get("usage", {})
            else:
                continue
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text_parts.extend(extract_text_parts_from_choice(choice))
            chunk_usage = data.get("usage")
            if isinstance(chunk_usage, Mapping):
                usage = dict(chunk_usage)
        return "".join(text_parts).strip(), usage

    if isinstance(result, str):
        parsed = parse_chat_completion_sse(result)
        return parsed.get("text", ""), parsed.get("usage", {})

    text = ""
    choices = getattr(result, "choices", None)
    if choices is None and isinstance(result, Mapping):
        choices = result.get("choices")
    if choices:
        first_choice = choices[0]
        direct_parts = extract_text_parts_from_choice(first_choice)
        if direct_parts:
            text = "".join(direct_parts)
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, Mapping):
            message = first_choice.get("message")
        if (not text) and message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, Mapping):
                content = message.get("content")
            text = "".join(flatten_chat_content_text(content))
    usage = getattr(result, "usage", None)
    if usage is None and isinstance(result, Mapping):
        usage = result.get("usage", {})
    usage = usage or {}
    return text.strip(), usage


def parse_chat_completion_http_payload(payload_text: str, content_type: str = "") -> tuple[str, Any, str]:
    text = payload_text or ""
    lowered_content_type = (content_type or "").lower()
    if "text/event-stream" in lowered_content_type or text.lstrip().startswith("data:"):
        parsed = parse_chat_completion_sse(text)
        return parsed.get("text", ""), parsed.get("usage", {}), text

    data = json.loads(text)
    reply = extract_text(data)[0]
    usage = data.get("usage", {}) if isinstance(data, Mapping) else {}
    return reply, usage, text


def response_text_safe(response: httpx.Response | None) -> str | None:
    if response is None:
        return None
    try:
        return response.text
    except httpx.ResponseNotRead:
        response.read()
        return response.text


def post_chat_completion_raw(url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[str, Any, str, str]:
    try:
        response = httpx.post(url, headers=headers, json=body, timeout=60.0)
        response.raise_for_status()
        text, usage, raw_payload = parse_chat_completion_http_payload(
            response.text,
            response.headers.get("content-type", ""),
        )
        content_type = response.headers.get("content-type", "")
        if text or body.get("stream") is True or "text/event-stream" not in content_type.lower():
            return text, usage, raw_payload, content_type
    except httpx.HTTPStatusError:
        if body.get("stream") is True:
            raise

    retry_body = dict(body)
    retry_body["stream"] = True
    retry_body["stream_options"] = {"include_usage": True}
    with httpx.stream("POST", url, headers=headers, json=retry_body, timeout=60.0) as response:
        response.raise_for_status()
        raw_chunks = list(response.iter_text())
        raw_payload = "".join(raw_chunks)
        text, usage, _ = parse_chat_completion_http_payload(
            raw_payload,
            response.headers.get("content-type", ""),
        )
        return text, usage, raw_payload, response.headers.get("content-type", "")


def model_supports_reasoning_effort(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return (
        lowered.startswith("o3")
        or lowered.startswith("o4")
        or "gpt-5" in lowered
        or lowered.startswith("openai/o3")
        or lowered.startswith("openai/o4")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("TAMUS_AI_CHAT_API_KEY") or os.getenv("TAMU_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL") or "https://chat-api.tamu.ai/api")
    parser.add_argument("--model", default="protected.gemini-2.0-flash-lite")
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing TAMU API key. Set TAMUS_AI_CHAT_API_KEY or TAMU_API_KEY.")

    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: TAMU API OK"}],
        "stream": False,
        "max_tokens": args.max_tokens,
    }
    if args.reasoning_effort and model_supports_reasoning_effort(args.model):
        body["reasoning_effort"] = args.reasoning_effort

    print("Request body:")
    print(json.dumps(body, indent=2))
    try:
        if args.base_url:
            url = args.base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            text, usage, raw_payload, content_type = post_chat_completion_raw(url, headers, body)
        else:
            client = OpenAI(api_key=args.api_key, base_url=args.base_url)
            result = client.chat.completions.create(**body)
            text, usage = extract_text(result)
            raw_payload = None
            content_type = None
        print("")
        print("Response summary:")
        print(json.dumps({
            "model": args.model,
            "reply": text,
            "usage": usage,
        }, indent=2))
        if not text:
            print("")
            print("Debug:")
            print(json.dumps({
                "content_type": content_type,
                "raw_response": raw_payload,
            }, indent=2))
        return 0
    except httpx.HTTPStatusError as exc:
        print("")
        print("HTTPStatusError", file=sys.stderr)
        print(json.dumps({
            "status_code": exc.response.status_code,
            "response": response_text_safe(exc.response),
        }, indent=2), file=sys.stderr)
        return 1
    except APIStatusError as exc:
        print("")
        print("APIStatusError", file=sys.stderr)
        print(json.dumps({
            "status_code": exc.status_code,
            "response": exc.response.text if exc.response is not None else None,
        }, indent=2), file=sys.stderr)
        return 1
    except APIError as exc:
        print("")
        print("APIError", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
