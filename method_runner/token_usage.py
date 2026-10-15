# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COMPONENTS = ("actor", "analyzer", "repair_memory_curator", "positive_skill_curator")


@dataclass
class TokenUsage:
    n_requests: int | None = None
    n_input_tokens: int | None = None
    n_cache_tokens: int | None = None
    n_output_tokens: int | None = None
    cost_usd: float | None = None
    source: str = "unknown"


def zero_usage() -> TokenUsage:
    return TokenUsage(0, 0, 0, 0, 0.0, "deterministic_python")


MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-opus-4.5": {"input": 5.0, "output": 25.0, "cache": 0.5},
    "claude-opus-4.6": {"input": 5.0, "output": 25.0, "cache": 0.5},
    "claude-sonnet-4.5": {"input": 3.0, "output": 15.0, "cache": 0.3},
    "claude-sonnet-4.6": {"input": 3.0, "output": 15.0, "cache": 0.3},
}


def non_cached_input_tokens(input_tokens: int | None, cache_tokens: int | None) -> int | None:
    if input_tokens is None:
        return None
    return max(int(input_tokens) - int(cache_tokens or 0), 0)


def estimate_cost_usd(model_name: str | None, input_tokens: int | None, cache_tokens: int | None, output_tokens: int | None) -> float | None:
    pricing = MODEL_PRICING_USD_PER_MILLION.get((model_name or "").strip().casefold().split("/", 1)[-1])
    if pricing is None:
        return None
    non_cached = non_cached_input_tokens(input_tokens, cache_tokens) or 0
    return round((non_cached * pricing["input"] + (cache_tokens or 0) * pricing["cache"] + (output_tokens or 0) * pricing["output"]) / 1_000_000, 6)


def _sum_optional(values: Any) -> int | float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def sum_token_usage(usages: list[TokenUsage], *, source: str = "aggregate") -> TokenUsage:
    return TokenUsage(
        n_requests=_sum_optional(usage.n_requests for usage in usages),
        n_input_tokens=_sum_optional(usage.n_input_tokens for usage in usages),
        n_cache_tokens=_sum_optional(usage.n_cache_tokens for usage in usages),
        n_output_tokens=_sum_optional(usage.n_output_tokens for usage in usages),
        cost_usd=_sum_optional(usage.cost_usd for usage in usages),
        source=source,
    )


def token_usage_dict(usage: TokenUsage) -> dict[str, Any]:
    return asdict(usage)


def token_usage_from_trial_dir(trial_dir: Path, model_name: str | None = None) -> TokenUsage:
    try:
        agent_result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8")).get("agent_result")
    except (OSError, json.JSONDecodeError):
        agent_result = None
    if not isinstance(agent_result, dict):
        return TokenUsage(source="harbor_usage_unavailable")
    metadata = agent_result.get("metadata") or {}
    usage = TokenUsage(
        n_requests=metadata.get("llm_requests", metadata.get("n_episodes", agent_result.get("n_requests"))),
        n_input_tokens=agent_result.get("n_input_tokens"),
        n_cache_tokens=agent_result.get("n_cache_tokens"),
        n_output_tokens=agent_result.get("n_output_tokens"),
        cost_usd=agent_result.get("cost_usd"),
        source="harbor_agent_context",
    )
    if usage.cost_usd is None:
        usage.cost_usd = estimate_cost_usd(model_name, usage.n_input_tokens, usage.n_cache_tokens, usage.n_output_tokens)
    return usage
