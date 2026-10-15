# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
import openai
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.metric import UsageInfo


def _uses_max_completion_tokens(model_name: str) -> bool:
    return model_name.strip().casefold().startswith(("gpt-5", "o1", "o3", "o4"))


class AzureOpenAILLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        *,
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        api_base: str | None = None,
        api_timeout_sec: float = 300.0,
        api_max_retries: int = 2,
        api_empty_response_retries: int = 3,
        reasoning_effort: str | None = None,
        max_input_tokens: int = 200000,
        max_output_tokens: int = 8192,
        **_: Any,
    ) -> None:
        super().__init__()
        self._model_name = model_name.split("/", 1)[-1]
        self._api_key_env = api_key_env
        self._api_base = api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self._api_base:
            raise RuntimeError("Azure endpoint is not set; define AZURE_OPENAI_ENDPOINT in .env")
        self._api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"
        self._api_timeout_sec = api_timeout_sec
        self._api_max_retries = api_max_retries
        self._api_empty_response_retries = api_empty_response_retries
        self._reasoning_effort = reasoning_effort
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens

    def get_model_context_limit(self) -> int:
        return self._max_input_tokens

    def get_model_output_limit(self) -> int | None:
        return self._max_output_tokens

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [*(message_history or []), {"role": "user", "content": prompt}]
        return await asyncio.to_thread(self._call_sync, messages, logging_path, kwargs)

    def _call_sync(self, messages: list[dict[str, Any]], logging_path: Path | None, kwargs: dict[str, Any]) -> LLMResponse:
        api_key = os.getenv(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"{self._api_key_env} is not set")
        client = openai.AzureOpenAI(
            azure_endpoint=self._api_base,
            api_key=api_key,
            api_version=self._api_version,
            timeout=self._api_timeout_sec,
            max_retries=self._api_max_retries,
        )
        request: dict[str, Any] = {"model": self._model_name, "messages": messages}
        reasoning_model = _uses_max_completion_tokens(self._model_name)
        if self._reasoning_effort not in {None, "none"} and reasoning_model:
            request["reasoning_effort"] = self._reasoning_effort
        limit = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if limit is not None:
            request["max_completion_tokens" if reasoning_model else "max_tokens"] = limit
        prompt_tokens = completion_tokens = cache_tokens = 0
        attempts = self._api_empty_response_retries + 1
        for attempt in range(attempts):
            response = client.chat.completions.create(**request)
            if logging_path is not None:
                logging_path.parent.mkdir(parents=True, exist_ok=True)
                logging_path.write_text(json.dumps(response.model_dump(mode="json"), indent=2), encoding="utf-8")
            usage = response.usage
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            cache_tokens += int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
            message = response.choices[0].message if response.choices else None
            content = getattr(message, "content", None) or ""
            if content.strip():
                return LLMResponse(
                    content=content,
                    reasoning_content=getattr(message, "reasoning_content", None),
                    model_name=response.model or self._model_name,
                    usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cache_tokens=cache_tokens, cost_usd=0.0),
                )
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Azure API returned no assistant content for model {self._model_name} after {attempts} attempts")
