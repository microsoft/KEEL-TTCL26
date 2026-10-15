# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
from typing import Any
from harbor.agents.terminus_2.terminus_json_plain_parser import ParseResult, ParsedCommand, TerminusJSONPlainParser

_FENCED_JSON_RE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_RECOVERY_KEY_RE = re.compile(
    r"(?P<prefix>^|\r?\n|\\n|[\{\[,])[ \t]*"
    r'\\?"(?P<key>analysis|plan|commands|keystrokes|duration|task_complete)'
    r'\\?"[ \t]*:',
    re.IGNORECASE,
)
_CONTROL_KEYS_RE = re.compile(r"(?:C|M|S)-(?:[A-Za-z]|Space|Enter)", re.IGNORECASE)
_SINGLE_ESCAPED_NEWLINE_RE = re.compile(r"(?<!\\)\\n")
_DOUBLE_ESCAPED_NEWLINE_RE = re.compile(r"\\\\n")
_ESCAPE_RE = re.compile(r'\\(?:u([0-9A-Fa-f]{4})|(["\\/bfnrt]))')
_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _decode_string_body(body: str) -> str:
    try:
        value = json.loads(f'"{body}"', strict=False)
        if isinstance(value, str):
            return value
    except json.JSONDecodeError:
        pass
    return _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else _SIMPLE_ESCAPES[m.group(2)], body)


def _extract_string_between(response: str, start: int, end: int) -> str:
    raw = response[start:end].strip()
    raw = raw.removeprefix('\\"') if raw.startswith('\\"') else raw.removeprefix('"')
    raw = raw.rstrip().removesuffix(",").rstrip()
    raw = raw.removesuffix('\\"') if raw.endswith('\\"') else raw.removesuffix('"')
    return _decode_string_body(raw)


def _recover_schema_fields(response: str) -> dict[str, Any] | None:
    tokens = [(m.group("key").lower(), m.start(), m.end()) for m in _RECOVERY_KEY_RE.finditer(response)]

    def first_after(key: str, offset: int) -> tuple[str, int, int] | None:
        return next((token for token in tokens if token[0] == key and token[1] >= offset), None)

    analysis_token = first_after("analysis", 0)
    if analysis_token is None:
        return None
    plan_token = first_after("plan", analysis_token[2])
    commands_token = first_after("commands", (plan_token or analysis_token)[2])
    if commands_token is None:
        return None
    analysis = _extract_string_between(response, analysis_token[2], (plan_token or commands_token)[1])
    plan = _extract_string_between(response, plan_token[2], commands_token[1]) if plan_token else ""
    task_token = first_after("task_complete", commands_token[2])
    commands_end = task_token[1] if task_token else len(response)
    command_tokens = [t for t in tokens if commands_token[2] <= t[1] < commands_end and t[0] in {"keystrokes", "duration"}]
    commands: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(command_tokens):
        key, _, keystrokes_end = command_tokens[cursor]
        cursor += 1
        if key != "keystrokes":
            continue
        keystrokes_stop = commands_end
        duration = 1.0
        if cursor < len(command_tokens):
            next_key, next_start, next_end = command_tokens[cursor]
            keystrokes_stop = next_start
            if next_key == "duration":
                duration_end = command_tokens[cursor + 1][2] if cursor + 1 < len(command_tokens) else commands_end
                number = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", response[next_end:duration_end])
                if number:
                    duration = float(number.group(0))
                cursor += 1
        commands.append({"keystrokes": _extract_string_between(response, keystrokes_end, keystrokes_stop), "duration": duration})
        if len(commands) > 64:
            return None
    if not commands and not re.search(r"\[\s*\]", response[commands_token[2] : commands_end]):
        return None
    task_complete = False
    if task_token:
        value = re.search(r"\b(true|false|yes|no|1|0)\b", response[task_token[2] :], re.I)
        task_complete = bool(value) and value.group(1).lower() in {"true", "yes", "1"}
    return {"analysis": analysis, "plan": plan, "commands": commands, "task_complete": task_complete}


class TolerantTerminusJSONParser(TerminusJSONPlainParser):
    def parse_response(self, response: str) -> ParseResult:
        match = _FENCED_JSON_RE.fullmatch(response)
        cleaned = match.group("body").strip() if match else response
        result = super().parse_response(cleaned)
        if not result.error:
            return result
        data = _recover_schema_fields(cleaned)
        if data is None:
            return result
        recovered = self._result_from_recovered(data)
        if recovered.error:
            return result
        warning = "AUTO-CORRECTED: recovered malformed Terminus JSON structure" + (" and removed Markdown fence" if match else "")
        recovered.warning = self._combine_warnings(warning, recovered.warning)
        return recovered

    def _result_from_recovered(self, data: dict[str, Any]) -> ParseResult:
        warnings: list[str] = []
        validation_error = self._validate_json_structure(data, json.dumps(data, ensure_ascii=False), warnings)
        if validation_error:
            return ParseResult([], False, validation_error, "", "", "")
        commands, parse_error = self._parse_commands(data["commands"], warnings)
        warning_text = "- " + "\n- ".join(warnings) if warnings else ""
        if parse_error:
            return ParseResult([], False, parse_error, warning_text, data["analysis"], data["plan"])
        return ParseResult(commands, bool(data.get("task_complete", False)), "", warning_text, data["analysis"], data["plan"])

    def _parse_commands(self, commands_data: list[dict[str, Any]], warnings: list[str]) -> tuple[list[ParsedCommand], str]:
        commands, error = super()._parse_commands(commands_data, warnings)
        if error:
            return commands, error
        for index, command in enumerate(commands):
            escaped_breaks = len(_SINGLE_ESCAPED_NEWLINE_RE.findall(command.keystrokes))
            if command.keystrokes.count("\n") > 1 or escaped_breaks < 2:
                continue
            sentinel = "\0LITERAL_NEWLINE\0"
            repaired = _DOUBLE_ESCAPED_NEWLINE_RE.sub(sentinel, command.keystrokes)
            repaired = _SINGLE_ESCAPED_NEWLINE_RE.sub("\n", repaired).replace(sentinel, "\\n")
            commands[index] = ParsedCommand(keystrokes=repaired, duration=command.duration)
            warnings.append(f"Command {index + 1}: decoded double-escaped line breaks")
        for index, command in enumerate(commands[:-1]):
            if command.keystrokes.endswith("\n") or _CONTROL_KEYS_RE.fullmatch(command.keystrokes.strip()):
                continue
            commands[index] = ParsedCommand(keystrokes=command.keystrokes + "\n", duration=command.duration)
            warnings.append(f"Command {index + 1}: appended missing newline before next command")
        return commands, ""
