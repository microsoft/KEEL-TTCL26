# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from analysis.skill_usage_parser import analyze_skill_usage, load_trajectory

READ_COMMAND_RE = re.compile(r"\b(cat|sed|head|tail|less|more|type)\b")
EXPLICIT_SKILL_READ_RE = re.compile(r"(?:^|[\s\"'])/root/\.agents/skills/(?P<name>[A-Za-z0-9_.-]+)/SKILL\.md(?:$|[\s\"'])")
NATIVE_SKILL_LOADED_RE = re.compile(r'Skill\s+"(?P<name>[A-Za-z0-9_.-]+)"\s+loaded successfully\.', re.IGNORECASE)
WILDCARD_SKILL_READ_RE = re.compile(r"/root/\.agents/skills/\*/SKILL\.md(?:$|[\s\"'])")


def _message_texts(step: dict[str, Any]) -> list[str]:
    message = step.get("message")
    if isinstance(message, str):
        return [message]
    return [value for value in message.values() if isinstance(value, str)] if isinstance(message, dict) else []


def _inspected_skills(trajectory: dict[str, Any], available: list[str]) -> tuple[set[str], int]:
    names: set[str] = set()
    calls = 0
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for text in _message_texts(step):
            loads = [match.group("name") for match in NATIVE_SKILL_LOADED_RE.finditer(text)]
            names.update(loads)
            calls += len(loads)
            for line in text.splitlines():
                command = line.strip()
                if command.startswith("$") and READ_COMMAND_RE.search(command):
                    reads = [match.group("name") for match in EXPLICIT_SKILL_READ_RE.finditer(command)]
                    names.update(reads)
                    calls += len(reads)
        if len(available) == 1:
            for call in step.get("tool_calls") or []:
                arguments = call.get("arguments") if isinstance(call, dict) else None
                if isinstance(arguments, dict) and any(
                    isinstance(value, str) and READ_COMMAND_RE.search(value) and WILDCARD_SKILL_READ_RE.search(value) for value in arguments.values()
                ):
                    names.update(available)
                    calls += 1
    return names, calls


def collect_skill_telemetry(trial_dir: Path, *, available_skills: list[str], availability_metadata: dict[str, Any]) -> dict[str, Any]:
    trajectory = load_trajectory(trial_dir / "agent" / "trajectory.json") or {"steps": []}
    stats = analyze_skill_usage(trajectory)
    available = sorted(set(available_skills))
    inspected, read_calls = _inspected_skills(trajectory, available)
    inspected |= {name for name in stats.skills_used if re.fullmatch(r"[A-Za-z0-9_.-]+", name)}
    used = sorted(set(available) & inspected)
    selected = sorted(set(availability_metadata.get("recommended_skills") or availability_metadata.get("selected_skills") or []))
    used_selected = sorted(set(selected) & inspected)
    telemetry = {
        "available_skills": available,
        "available_skill_count": len(available),
        "eligible_skills": sorted(set(availability_metadata.get("eligible_skills") or available)),
        "selected_skills": selected,
        "selected_skill_count": len(selected),
        "routing_policy": availability_metadata.get("routing_policy", "all_active_skills"),
        "used_skills": used,
        "used_available_skills": used,
        "skill_read_calls": max(stats.skill_use_calls, read_calls),
        "skill_available": bool(available),
        "skill_selected": bool(selected),
        "skill_used": bool(used),
        "selected_skill_consumed": bool(used_selected),
    }
    telemetry["eligible_skill_count"] = len(telemetry["eligible_skills"])
    output_path = trial_dir / "agent" / "skill_usage.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
    return telemetry
