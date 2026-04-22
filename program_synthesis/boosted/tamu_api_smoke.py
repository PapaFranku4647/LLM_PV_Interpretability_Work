from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_SYNTHESIS_DIR = os.path.dirname(CURRENT_DIR)
if PROGRAM_SYNTHESIS_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_SYNTHESIS_DIR)

import runner as base_runner  # noqa: E402


def _extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TAMU/Azure OpenAI smoke test.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5"), help="Model or Azure deployment name.")
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat_completions"],
        default=os.getenv("API_MODE", "responses"),
        help="Which endpoint family to smoke-test.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Max output tokens.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds.")
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("REASONING_EFFORT", "high"),
        help="Responses API reasoning effort: low, medium, high, or xhigh.",
    )
    parser.add_argument(
        "--prompt",
        default='Return a compact JSON object exactly like {"ok": true}.',
        help="Prompt used for the smoke test.",
    )
    parser.add_argument("--api-key", help="API key override.")
    parser.add_argument("--api-base-url", help="OpenAI-compatible base URL override.")
    parser.add_argument("--azure-endpoint", help="Azure/TAMU endpoint override.")
    parser.add_argument("--api-version", help="Azure/TAMU API version override.")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    cfg = base_runner.Config()
    cfg.model = args.model
    cfg.api_mode = args.api_mode
    cfg.per_call_timeout_s = args.timeout
    cfg.max_output_tokens = args.max_output_tokens
    cfg.reasoning_effort = args.reasoning_effort
    if args.api_key is not None:
        cfg.api_key = args.api_key
        cfg.api_key_explicit = True
    if args.api_base_url is not None:
        cfg.api_base_url = args.api_base_url
    if args.azure_endpoint is not None:
        cfg.azure_endpoint = args.azure_endpoint
    if args.api_version is not None:
        cfg.api_version = args.api_version

    try:
        client, client_type = base_runner.create_openai_client(cfg)
    except SystemExit as exc:
        print(json.dumps({"skipped": True, "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 0

    request_model = base_runner.resolve_openai_request_model(cfg)
    response_model = request_model
    text = ""
    usage: Dict[str, Any] = {}
    try:
        if args.api_mode == "responses":
            response_kwargs = {
                "model": request_model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": args.prompt}]}],
                "max_output_tokens": args.max_output_tokens,
            }
            if args.reasoning_effort:
                response_kwargs["reasoning"] = {"effort": args.reasoning_effort}
            responses_api = getattr(client, "responses", None)
            create_fn = getattr(responses_api, "create", None)
            if callable(create_fn):
                response = await asyncio.wait_for(
                    create_fn(**response_kwargs),
                    timeout=args.timeout,
                )
            else:
                settings = base_runner.resolve_openai_client_settings(cfg)
                if not settings["use_azure"]:
                    raise RuntimeError(
                        "The installed OpenAI SDK does not expose responses.create for this client."
                    )
                try:
                    import litellm
                except ImportError as exc:
                    raise RuntimeError(
                        "The installed OpenAI SDK does not expose Azure responses.create and LiteLLM is not installed."
                    ) from exc
                litellm_kwargs = dict(response_kwargs)
                litellm_kwargs["model"] = f"azure/{request_model}"
                litellm_kwargs["api_key"] = settings["api_key"]
                litellm_kwargs["api_base"] = settings["azure_endpoint"].rstrip("/")
                litellm_kwargs["api_version"] = settings["api_version"]
                response = await asyncio.wait_for(
                    asyncio.to_thread(litellm.responses, **litellm_kwargs),
                    timeout=args.timeout,
                )
            response_model = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
            response_model = response_model or request_model
            text = base_runner.Runner._extract_text_from_responses_response(response)
            usage_obj = response.get("usage", {}) if isinstance(response, dict) else getattr(response, "usage", {})
            usage = base_runner.normalize_usage(usage_obj)
        else:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=request_model,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_completion_tokens=args.max_output_tokens,
                ),
                timeout=args.timeout,
            )
            response_model = getattr(response, "model", None) or request_model
            text = _extract_chat_text(response)
            usage = base_runner.normalize_usage(getattr(response, "usage", {}))

        print(
            json.dumps(
                {
                    "ok": bool(text),
                    "client_type": client_type,
                    "api_mode": args.api_mode,
                    "reasoning_effort": args.reasoning_effort,
                    "requested_model": args.model,
                    "request_model": request_model,
                    "model": response_model,
                    "usage": usage,
                    "text_preview": text[:200],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if text else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "client_type": client_type,
                    "api_mode": args.api_mode,
                    "reasoning_effort": args.reasoning_effort,
                    "requested_model": args.model,
                    "request_model": request_model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            maybe_awaitable = close_fn()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
