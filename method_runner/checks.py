# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import ast
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    name: str
    status: str
    message: str = ""
    first_assertion: str | None = None
    expected: str | None = None
    actual: str | None = None


@dataclass
class VerifierEvidence:
    reward: float
    total: int
    passed: int
    failed: int
    checks: dict[str, CheckResult] = field(default_factory=dict)
    infrastructure_error: str | None = None


@dataclass
class CheckPartition:
    passed: list[str]
    failed: list[str]
    unresolved: list[str]
    infrastructure_error: list[str]


ASSERT_RE = re.compile(r"AssertionError:\s*(.+)")
EXPECTED_GOT_RE = re.compile(r"expected\s+(.+?),\s*got\s+(.+)", re.IGNORECASE)
PYTEST_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<status>failed|passed|skipped|xfailed|xpassed|error|errors)\b", re.IGNORECASE)
VERIFIER_DEPENDENCY_ERROR_RE = re.compile(r"ImportError:\s+Missing optional dependency\s+['\"][^'\"]+['\"]", re.IGNORECASE)
VERIFIER_RUNTIME_ERROR_RE = re.compile(r"INTERNALERROR>|Unknown option --recalculate", re.IGNORECASE)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def reward_from_result(result: dict[str, Any]) -> float:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    try:
        return float(rewards.get("reward", result.get("reward", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def iter_test_functions(module: ast.Module):
    for node in module.body:
        for fn in node.body if isinstance(node, ast.ClassDef) else [node]:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test_"):
                yield fn


def literal_eval_or_none(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [literal_eval_or_none(item, constants) for item in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict):
        return {literal_eval_or_none(k, constants): literal_eval_or_none(v, constants) for k, v in zip(node.keys, node.values, strict=True) if k is not None}
    if isinstance(node, ast.Lambda):
        return "<lambda>"
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def module_constants(module: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value_node = node.target.id, node.value
        else:
            continue
        value = literal_eval_or_none(value_node, constants) if value_node is not None else None
        if value is not None:
            constants[name] = value
    return constants


def is_pytest_parametrize(func: ast.AST) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr != "parametrize":
        return False
    value = func.value
    if isinstance(value, ast.Attribute) and value.attr == "mark":
        return isinstance(value.value, ast.Name) and value.value.id == "pytest"
    return isinstance(value, ast.Name) and value.id == "mark"


def param_id(value: Any, arg_name: str, index: int) -> str:
    return str(value) if value is None or isinstance(value, (str, int, float, bool)) else f"{arg_name}{index}"


def expand_parametrized_test_names(node: ast.FunctionDef | ast.AsyncFunctionDef, constants: dict[str, Any]) -> list[str]:
    groups: list[list[str]] = []
    for call in node.decorator_list:
        if not isinstance(call, ast.Call) or not is_pytest_parametrize(call.func) or len(call.args) < 2:
            continue
        arg_names_value = literal_eval_or_none(call.args[0], constants)
        arg_names = [part.strip() for part in arg_names_value.split(",")] if isinstance(arg_names_value, str) else []
        values = literal_eval_or_none(call.args[1], constants)
        if not isinstance(values, (list, tuple)):
            continue
        explicit_ids = next((literal_eval_or_none(kw.value, constants) for kw in call.keywords if kw.arg == "ids"), None)
        explicit_ids = list(explicit_ids) if isinstance(explicit_ids, (list, tuple)) else []
        case_ids = []
        for index, value in enumerate(values):
            if index < len(explicit_ids):
                case_ids.append(str(explicit_ids[index]))
            elif isinstance(value, (list, tuple)) and len(arg_names) > 1:
                case_ids.append("-".join(param_id(item, arg_names[i] if i < len(arg_names) else "case", index) for i, item in enumerate(value)))
            else:
                case_ids.append(param_id(value, arg_names[0] if arg_names else "case", index))
        if case_ids:
            groups.append(case_ids)
    if not groups:
        return []
    return [f"{node.name}[{'-'.join(parts)}]" for parts in itertools.product(*reversed(groups))]


def can_skip(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "skip"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "pytest"
        for child in ast.walk(node)
    )


def discover_tests(task_dir: Path) -> tuple[list[str], list[str]]:
    modules: list[tuple[Path, ast.Module]] = []
    for test_path in sorted((task_dir / "tests").glob("test*.py")):
        try:
            modules.append((test_path, ast.parse(test_path.read_text(encoding="utf-8"))))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
    inner = [item for item in modules if item[0].name not in {"test_output.py", "test.py"}]
    if any(next(iter_test_functions(module), None) for _, module in inner):
        modules = inner
    names: list[str] = []
    skippable: list[str] = []
    for _, module in modules:
        constants = module_constants(module)
        for node in iter_test_functions(module):
            expanded = expand_parametrized_test_names(node, constants) or [node.name]
            names.extend(expanded)
            if can_skip(node):
                skippable.extend(expanded)
    return list(dict.fromkeys(names)), list(dict.fromkeys(skippable))


def extract_outcome_names(stdout: str, outcome: str) -> list[str]:
    pattern = re.compile(rf"\b{outcome}\s+(?P<node>.+?)(?:\s+-\s+.*)?$")
    names: list[str] = []
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match is None or "::" not in match.group("node"):
            continue
        name = match.group("node").rsplit("::", 1)[-1].strip()
        if name.startswith("test_") and name not in names:
            names.append(name)
    return names


def extract_inner_pytest_counts(stdout: str) -> dict[str, int]:
    candidates: list[dict[str, int]] = []
    for line in stdout.splitlines():
        counts: dict[str, int] = {}
        for match in PYTEST_COUNT_RE.finditer(line):
            status = "error" if match.group("status").casefold() == "errors" else match.group("status").casefold()
            counts[status] = counts.get(status, 0) + int(match.group("count"))
        if counts.keys() & {"passed", "failed", "error"}:
            candidates.append(counts)
    if not candidates:
        return {}
    return max(enumerate(candidates), key=lambda item: (sum(item[1].values()), item[0]))[1]


def extract_failure_details(stdout: str, test_name: str) -> tuple[str | None, str | None, str | None]:
    start = stdout.find(test_name)
    chunk = stdout[start : start + 3500] if start >= 0 else stdout[:3500]
    first_assertion = expected = actual = None
    for line in chunk.splitlines():
        line = line.strip()
        if first_assertion is None and (match := ASSERT_RE.search(line)):
            first_assertion = match.group(1).strip()
        if (expected is None or actual is None) and (match := EXPECTED_GOT_RE.search(line)):
            expected, actual = match.group(1).strip(), match.group(2).strip()
    return first_assertion, expected, actual


def parse_verifier_evidence(trial_dir: Path, task_dir: Path | None = None) -> VerifierEvidence:
    result = load_json(trial_dir / "result.json")
    reward = reward_from_result(result)
    infrastructure_error = None
    exc = result.get("exception_info")
    if isinstance(exc, dict) and exc:
        infrastructure_error = str(exc.get("exception_type") or exc.get("exception_message") or exc)
    stdout_path = trial_dir / "verifier" / "test-stdout.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    summary = (load_json(trial_dir / "verifier" / "ctrf.json").get("results") or {}).get("summary") or {}
    total, passed, failed = (int(summary.get(key) or 0) for key in ("tests", "passed", "failed"))
    failed_names = extract_outcome_names(stdout, "FAILED")
    error_names = extract_outcome_names(stdout, "ERROR")
    inner_counts = extract_inner_pytest_counts(stdout)
    all_names, skippable_names = discover_tests(task_dir) if task_dir else ([], [])
    observed = [name for name in [*failed_names, *error_names] if name != "test_legacy_pytest_suite"]
    all_names = list(dict.fromkeys([*all_names, *observed]))
    if all_names and not (stdout.strip() or summary) and infrastructure_error:
        checks = {name: CheckResult(name, "unknown", infrastructure_error) for name in all_names}
        return VerifierEvidence(reward, len(all_names), 0, 0, checks, infrastructure_error)
    known = set(all_names)
    failed_names = [name for name in failed_names if name in known]
    error_names = [name for name in error_names if name in known]
    failed_or_error = set(failed_names) | set(error_names)
    complete_inner_run = bool(all_names) and (reward >= 1.0 or sum(inner_counts.values()) >= len(all_names))
    unattributed_errors = max(0, inner_counts.get("error", 0) - len(set(error_names)))
    if not infrastructure_error:
        if dependency_error := VERIFIER_DEPENDENCY_ERROR_RE.search(stdout):
            infrastructure_error = "Verifier environment dependency error: " + dependency_error.group(0)
        elif runtime_error := VERIFIER_RUNTIME_ERROR_RE.search(stdout):
            infrastructure_error = "Verifier runtime error: " + runtime_error.group(0)
        elif unattributed_errors:
            infrastructure_error = f"Inner verifier suite reported {unattributed_errors} unattributed collection or execution error(s)."
    if all_names:
        total, failed = len(all_names), len(failed_or_error)
        passed = (total if reward >= 1.0 else min(inner_counts.get("passed", 0), max(0, total - failed))) if complete_inner_run else 0
    skipped_count = inner_counts.get("skipped", 0) + inner_counts.get("xfailed", 0) if complete_inner_run and reward < 1.0 else 0
    skipped_candidates = [name for name in [*skippable_names, *all_names] if name in known and name not in failed_or_error]
    skipped_names = set(list(dict.fromkeys(skipped_candidates))[:skipped_count])
    checks: dict[str, CheckResult] = {}
    for name in all_names:
        if name in failed_or_error:
            first_assertion, expected, actual = extract_failure_details(stdout, name)
            if name in error_names and infrastructure_error:
                checks[name] = CheckResult(name, "error", infrastructure_error, first_assertion, expected, actual)
            else:
                fallback = "The task check raised while validating the generated artifact." if name in error_names else ""
                checks[name] = CheckResult(name, "failed", first_assertion or fallback, first_assertion, expected, actual)
        elif name in skipped_names:
            checks[name] = CheckResult(name, "unknown", "The inner verifier skipped this check.")
        elif complete_inner_run:
            checks[name] = CheckResult(name, "passed")
        else:
            checks[name] = CheckResult(name, "unknown", "The verifier did not prove that this check completed.")
    return VerifierEvidence(reward, total, passed, failed, checks, infrastructure_error)


def partition_checks(evidence: VerifierEvidence) -> CheckPartition:
    partition = CheckPartition([], [], [], [])
    for name in sorted(evidence.checks):
        status = evidence.checks[name].status
        if status == "passed":
            partition.passed.append(name)
        elif status == "failed":
            partition.failed.append(name)
        elif status == "error" or evidence.infrastructure_error:
            partition.infrastructure_error.append(name)
        else:
            partition.unresolved.append(name)
    return partition
