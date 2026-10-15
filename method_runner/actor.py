# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from harbor.models.job.config import JobConfig
from method_runner.analyzer import build_requirement_ledger

MAX_ACTOR_MEMORY_ITEMS = 1
MAX_ACTOR_MEMORY_CHARS = 1_000
MAX_MEMORY_FIELD_CHARS = 180
MEMORY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")
MEMORY_STOPWORDS = {"and", "are", "for", "from", "into", "not", "only", "output", "sheet", "task", "that", "the", "their", "then", "this", "use", "when", "with"}
SEPARATOR = "\n\n---\n\n"

LEDGER_HEADER = (
    "# Exact Requirement Ledger\n\n"
    "Treat every item below as a current-task obligation. Resolve any "
    "task-specific values from the current instruction and files; do not "
    "replace them with values from skills or repair memory.\n\n"
)
SKILL_PREFLIGHT = (
    "# Mandatory Skill Preflight\n\n"
    "Active family skills are mounted at `/root/.agents/skills`. Before "
    "modifying any task artifact:\n\n"
    "1. Inspect the available skill names and descriptions.\n"
    "2. Invoke the native `skill` tool or read the complete `SKILL.md` "
    "for the most applicable candidate using its concrete path; do not "
    "use a wildcard `*/SKILL.md` path.\n"
    "3. Compare its applicability and exclusions with the current task.\n"
    "4. Apply only compatible instructions. If the inspected skill is "
    "not applicable, ignore it and continue from current task evidence.\n\n"
    "Do not begin artifact modification before completing this inspection. "
    "The current task instruction and artifacts always take priority."
)
NO_SKILLS = "# Shared Family Skills\n\nNo active family skill is available for this task. Solve it from the current instruction and artifacts."
REQUIREMENTS_SECTION = (
    "# Current Task Requirements and Mandatory Validation Gate\n\n"
    "- Before editing, build an exact requirement ledger covering output paths, "
    "schemas, literals, locations, transformations, preserved properties, and "
    "verification obligations.\n"
    "- Put the ledger and every injected prior-family `Verify` clause into the "
    "task tracker as concrete checks; reading a lesson without executing its "
    "check does not satisfy it.\n"
    "- This is the only Actor rollout for this task; validate the result carefully.\n"
    "- Search the local environment for authoritative task tests under `/tests`, "
    "`/root/tests`, `/workspace`, and task directories. Treat them as authoritative "
    "when available.\n"
    "- Run the narrowest applicable pytest, Python, Node, or shell verifier before "
    "finalizing. Run it directly without piping through `tail`, `grep`, or another "
    "command that can hide the verifier exit code. If it fails, repair the artifact "
    "and rerun it until it passes.\n"
    "- If a verifier file is a wrapper around another test module, inspect and run "
    "the underlying module directly so all assertions and failures are visible.\n"
    "- If no verifier is accessible, create `/tmp/skillflow_validate.py`, execute "
    "it, and make it reopen the saved artifact and assert exact literals, locations, "
    "types, formulas and cached values, schemas, and preservation constraints.\n"
    "- If verifier assertions expose exact cells, filenames, schemas, or values, satisfy those exact checks.\n"
    "- Never invent an exact title, header, schema key, cell location, conversion "
    "direction, or aggregation rule when it can be resolved from the instruction, "
    "source artifact, or local verifier.\n"
    "- For source-file metadata, distinguish an artifact location from its stored "
    "identifier; unless the schema explicitly requests an absolute path, store the "
    "source basename and validate it exactly.\n"
    "- For embedded Office workbooks, compare formula-mode and `data_only=True` "
    "reads. A required formula whose cached value becomes `None` is a validation "
    "failure: recalculate with an office engine or use a surgical OOXML edit that "
    "preserves existing formula caches.\n"
    "- For presentation layout language such as widened, bottom, centered, or "
    "single-line, validate geometry relative to the slide dimensions. Use at least "
    "90% of slide width for a widened single-line caption unless the task specifies "
    "another exact size.\n"
    "- Before finalizing, reopen the saved artifact and check it against every item "
    "in the requirement ledger; do not claim success from an in-memory object or "
    "visual inspection alone."
)


def _keywords(text: str) -> set[str]:
    return {word.casefold() for word in MEMORY_WORD_RE.findall(text) if word.casefold() not in MEMORY_STOPWORDS}


def _compact(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= MAX_MEMORY_FIELD_CHARS else text[: MAX_MEMORY_FIELD_CHARS - 1].rstrip() + "…"


def _support(item: dict[str, Any]) -> tuple[int, int]:
    return len(item.get("supporting_tasks") or []), int(item.get("last_task_position") or 0)


def _memory_block(item: dict[str, Any]) -> list[str]:
    when = f"- When: {_compact(item.get('trigger'))}"
    verify = f"  Verify: {_compact(item.get('verify'))}"
    if (item.get("lesson_type") or "diagnostic") == "diagnostic":
        observed = _compact(item.get("observed_failure") or item.get("failure_pattern"))
        inspect = _compact(item.get("inspection_action") or f"Check current evidence to determine whether this recurs: {observed}")
        return ["", when, f"  Observed: {observed}", f"  Inspect: {inspect}", verify]
    return ["", when, f"  Avoid: {_compact(item.get('failure_pattern'))}", f"  Do: {_compact(item.get('corrective_action') or item.get('avoid'))}", verify]


def select_repair_memory(repair_memory_path: Path, task_instruction: str) -> tuple[str, dict[str, Any]]:
    try:
        items = json.loads(repair_memory_path.with_suffix(".state.json").read_text(encoding="utf-8")).get("items") or []
    except (OSError, json.JSONDecodeError, AttributeError):
        items = []
    active = [item for item in items if isinstance(item, dict) and item.get("status") == "active"]
    task_keywords = _keywords(task_instruction)
    overlaps = {id(item): len(task_keywords & _keywords(str(item.get("trigger", "")))) for item in active}
    matched = sorted((item for item in active if overlaps[id(item)] > 0), key=lambda item: (overlaps[id(item)], *_support(item)), reverse=True)
    candidates = matched or sorted(active, key=_support, reverse=True)[:1]
    lines = ["## Relevant Prior Family Lessons"]
    injected_ids: list[str] = []
    suppressed_claim_ids: set[str] = set()
    for item in candidates:
        if len(injected_ids) >= MAX_ACTOR_MEMORY_ITEMS:
            break
        block = _memory_block(item)
        if len("\n".join(lines + block)) + 1 > MAX_ACTOR_MEMORY_CHARS:
            continue
        lines.extend(block)
        injected_ids.append(str(item.get("id") or item.get("title") or "unknown"))
        suppressed_claim_ids.update(str(claim_id) for claim_id in item.get("related_skill_claim_ids") or [] if str(claim_id))
    text = "\n".join(lines) + "\n" if injected_ids else ""
    metrics = {
        "available_items": len(active),
        "matched_items": len(matched),
        "injected_items": len(injected_ids),
        "injected_memory_ids": injected_ids,
        "suppressed_skill_claim_ids": sorted(suppressed_claim_ids),
    }
    return text, metrics


def build_augmented_instruction(base_instruction: str, memory_text: str, skill_routing: dict[str, Any]) -> str:
    ledger = "\n".join(f"{index}. [{item['type']}] {item['exact_requirement']}" for index, item in enumerate(build_requirement_ledger(base_instruction), start=1))
    ledger_section = LEDGER_HEADER + (ledger or "1. [general] Follow the complete current task instruction.")
    selected = list(skill_routing.get("recommended_skills") or skill_routing.get("selected_skills") or [])
    if skill_routing.get("available_skills") or selected:
        skills_section = SKILL_PREFLIGHT
        if selected:
            name = str(selected[0])
            record = next((item for item in skill_routing.get("routing_records") or [] if item.get("skill_name") == name), {})
            terms = ", ".join(record.get("matched_terms") or []) or "family applicability"
            description = str(record.get("description") or "Active family skill.").strip()
            skills_section += (
                f"\n\nHigh-confidence candidate to inspect first:\n\n- `{name}`: {description}\n"
                f"  Matched task terms: {terms}\n  Inspect: `/root/.agents/skills/{name}/SKILL.md`"
            )
    else:
        skills_section = NO_SKILLS
    sections = [base_instruction.rstrip(), ledger_section, skills_section]
    if memory_text:
        sections.append(memory_text.rstrip())
    return SEPARATOR.join(sections + [REQUIREMENTS_SECTION])


def windows_extended_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute.lstrip("\\"))
    return Path("\\\\?\\" + absolute)


def remove_tree(path: Path) -> None:
    def retry_with_write_access(function: Any, item: str, excinfo: BaseException) -> None:
        Path(item).chmod(0o700)
        function(item)

    shutil.rmtree(windows_extended_path(path), onexc=retry_with_write_access)


def copy_task_overlay(source_task: Path, overlay_task: Path, instruction: str) -> None:
    if overlay_task.exists():
        remove_tree(overlay_task)
    overlay_path = windows_extended_path(overlay_task)
    shutil.copytree(windows_extended_path(source_task), overlay_path, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (overlay_path / "instruction.md").write_text(instruction, encoding="utf-8")


def build_iteration_job_config(
    base_config: JobConfig, *, job_name: str, jobs_dir: Path, family_name: str, dataset_root: Path, overlay_task: Path, shared_skills_dir: Path
) -> JobConfig:
    data = base_config.model_dump()
    data.update(
        job_name=job_name,
        jobs_dir=str(jobs_dir),
        n_attempts=1,
        n_concurrent_trials=1,
        quiet=True,
        tasks=[{"path": str(overlay_task), "source": family_name}],
        datasets=[{"path": str(dataset_root), "n_tasks": 0}],
    )
    env = data.get("environment") or {}
    env["import_path"] = "libs.terminus_env.environments.shared_skills_env:SharedSkillsDockerEnvironment"
    env["kwargs"] = {
        **(env.get("kwargs") or {}),
        "project_template_dir": str(shared_skills_dir),
        "shared_skills_root": "s",
        "copy_task_skills": False,
    }
    data["environment"] = env
    return JobConfig.model_validate(data)
