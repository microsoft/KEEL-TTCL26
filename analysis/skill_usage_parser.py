# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILL_NAME_KEYS = ("skill", "skill_name", "name")
FILE_EDITOR_TOOL_NAMES = ("file_editor", "fileeditor")
FILE_EDITOR_READ_COMMANDS = ("read", "view")
SHELL_READ_COMMAND_PATTERN = re.compile(r"(?:^|[;&|]\s*|\b)(?:cat|sed|awk|grep|head|tail|less|more)\b", re.IGNORECASE)
SKILL_PATH_PATTERN = re.compile(r"(?:/|^)(?:\.claude/skills|\.codex/skills|\.agents/skills|skills)/([^/\\\"'\s]+)(?:/SKILL\.md|/|$)")


@dataclass(frozen=True)
class SkillUsageStats:
    skill_use_steps: int
    skill_use_calls: int
    skill_usage_rate: float
    skills_used: list[str]


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_string_values(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_string_values(nested)


def _normalize_skill_names(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {item.strip() for item in value if isinstance(item, str) and item.strip()}
    return set()


def extract_skill_names(arguments: dict[str, Any] | None) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    names: set[str] = set()
    for key in SKILL_NAME_KEYS:
        names |= _normalize_skill_names(arguments.get(key))
    if names:
        return sorted(names)
    for text in _iter_string_values(arguments):
        names.update(match.group(1).strip() for match in SKILL_PATH_PATTERN.finditer(text) if match.group(1).strip())
    return sorted(names)


def is_skill_usage_call(tool_call: dict[str, Any]) -> bool:
    function_name = tool_call.get("function_name") or tool_call.get("name") or ""
    name = function_name.strip().lower() if isinstance(function_name, str) else ""
    if name == "skill":
        return True
    arguments = tool_call.get("arguments")
    if name == "read":
        return bool(extract_skill_names(arguments))
    if name in FILE_EDITOR_TOOL_NAMES and isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip().lower()
        return command in FILE_EDITOR_READ_COMMANDS and bool(extract_skill_names(arguments))
    if not isinstance(arguments, dict):
        return False
    return any(SKILL_PATH_PATTERN.search(text) and SHELL_READ_COMMAND_PATTERN.search(text) for text in _iter_string_values(arguments))


def analyze_skill_usage(trajectory: dict[str, Any] | None) -> SkillUsageStats:
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list) or not steps:
        return SkillUsageStats(0, 0, 0.0, [])
    skill_use_steps = 0
    skill_use_calls = 0
    skills_used: set[str] = set()
    for step in steps:
        tool_calls = step.get("tool_calls") if isinstance(step, dict) else None
        if not isinstance(tool_calls, list):
            continue
        calls = [call for call in tool_calls if isinstance(call, dict) and is_skill_usage_call(call)]
        skill_use_calls += len(calls)
        skill_use_steps += bool(calls)
        for call in calls:
            skills_used.update(extract_skill_names(call.get("arguments")))
    return SkillUsageStats(skill_use_steps, skill_use_calls, round(skill_use_steps / len(steps), 4), sorted(skills_used))


def load_trajectory(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
