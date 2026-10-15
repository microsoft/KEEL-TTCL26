# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from pydantic import BaseModel
from method_runner.token_usage import TokenUsage, estimate_cost_usd, sum_token_usage

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass
class StructuredLLMRun:
    value: BaseModel | None
    token_usage: TokenUsage
    errors: list[str]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", stripped)).strip()
    try:
        payload = json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise
        payload, _ = json.JSONDecoder(strict=False).raw_decode(stripped[start:])
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be one JSON object")
    return payload


def usage_from_response(response: Any, source: str, model_name: str | None = None) -> TokenUsage:
    usage = response.usage
    input_tokens, cache_tokens, output_tokens = int(usage.prompt_tokens or 0), int(usage.cache_tokens or 0), int(usage.completion_tokens or 0)
    estimated_cost = estimate_cost_usd(model_name, input_tokens, cache_tokens, output_tokens)
    return TokenUsage(
        n_requests=1,
        n_input_tokens=input_tokens,
        n_cache_tokens=cache_tokens,
        n_output_tokens=output_tokens,
        cost_usd=estimated_cost if estimated_cost is not None else float(usage.cost_usd or 0.0),
        source=source,
    )


def call_structured_llm(
    *,
    client: Any,
    model_type: type[ModelT],
    system_prompt: str,
    prompt: str,
    log_prefix: Path,
    max_output_tokens: int,
    max_attempts: int,
    usage_source: str,
    validator: Callable[[ModelT], None] | None = None,
    payload_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    value_transform: Callable[[ModelT], ModelT] | None = None,
    failure_value_transform: Callable[[ModelT], ModelT] | None = None,
    model_name: str | None = None,
) -> StructuredLLMRun:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    usages: list[TokenUsage] = []
    errors: list[str] = []
    previous_output = ""
    request_count = 0
    last_validated_shape: ModelT | None = None

    def finish(value: ModelT | None) -> StructuredLLMRun:
        source = usage_source if value is not None else f"{usage_source}_failed"
        usage = sum_token_usage(usages, source=source) if usages else TokenUsage(source=source)
        usage.n_requests = request_count
        return StructuredLLMRun(value=value, token_usage=usage, errors=errors)

    for call_index in range(1, max_attempts + 1):
        current_prompt = prompt
        if errors:
            current_prompt += (
                "\n\nYour previous output was invalid. Correct it without changing the supplied evidence. Validation error:\n"
                + errors[-1]
                + "\n\nPREVIOUS OUTPUT:\n"
                + previous_output[:20_000]
            )
        try:
            request_count += 1
            response = asyncio.run(
                client.call(
                    current_prompt,
                    message_history=[{"role": "system", "content": system_prompt}],
                    logging_path=log_prefix.with_suffix(f".response_attempt_{call_index}.json"),
                    temperature=0,
                    max_tokens=max_output_tokens,
                )
            )
            usages.append(usage_from_response(response, usage_source, model_name))
            previous_output = response.content
            payload = extract_json_object(response.content)
            value = model_type.model_validate(payload_transform(payload) if payload_transform else payload)
            last_validated_shape = value
            if value_transform:
                value = value_transform(value)
            if validator:
                validator(value)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        return finish(value)
    if failure_value_transform and last_validated_shape is not None:
        try:
            fallback_value = failure_value_transform(last_validated_shape)
            if validator:
                validator(fallback_value)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        else:
            return finish(fallback_value)
    return finish(None)
