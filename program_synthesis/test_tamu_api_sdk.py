from __future__ import annotations

import argparse
import json
import sys

try:
    from program_synthesis.llm_client import (
        build_chat_completions_body,
        build_sync_client,
        call_llm_sync,
        resolve_api_key_from_env,
        resolve_api_version_from_env,
        resolve_azure_endpoint_from_env,
        resolve_default_model_from_env,
    )
except ModuleNotFoundError:
    from llm_client import (  # type: ignore
        build_chat_completions_body,
        build_sync_client,
        call_llm_sync,
        resolve_api_key_from_env,
        resolve_api_version_from_env,
        resolve_azure_endpoint_from_env,
        resolve_default_model_from_env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=resolve_api_key_from_env())
    parser.add_argument("--azure-endpoint", default=resolve_azure_endpoint_from_env() or "https://tamu-it-ae-ai-prod-prod-eastus2.openai.azure.com/")
    parser.add_argument("--api-version", default=resolve_api_version_from_env())
    parser.add_argument("--model", default=resolve_default_model_from_env("gpt-5.2-deep-learning-fundamentals"))
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing TAMU API key. Set TAMU_API_KEY or TAMUS_AI_CHAT_API_KEY.")

    client, _ = build_sync_client(
        args.api_key,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
    )
    body = build_chat_completions_body(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with exactly: TAMU API OK"}],
        max_output_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        stream=False,
        azure_endpoint=args.azure_endpoint,
    )

    print("Request body:")
    print(json.dumps(body, indent=2))

    try:
        result = call_llm_sync(
            client=client,
            api_mode="chat_completions",
            body=body,
            timeout=60.0,
            azure_endpoint=args.azure_endpoint,
        )
        print("")
        print("Response summary:")
        print(json.dumps({
            "deployment": args.model,
            "reply": result.text,
            "usage": result.usage,
            "returned_model": result.model,
        }, indent=2))
        return 0
    except Exception as exc:
        print("")
        print("SDK call failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
