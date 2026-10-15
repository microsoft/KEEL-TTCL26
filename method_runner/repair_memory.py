# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from libs.azure_agents.azure_llm import AzureOpenAILLM
from method_runner.llm_json import call_structured_llm
from method_runner.token_usage import TokenUsage, token_usage_dict, zero_usage

MEMORY_CURATOR_SYSTEM_PROMPT = """You are the Family Repair Memory Curator in a
verifier-guided lifelong-learning experiment. You do not operate a terminal.

Produce zero to two compact family-level lessons from the supplied Analyzer
failure evidence.

Each lesson must be one of two types:
- ``diagnostic`` (the default): contain ``when``, ``observed_failure``,
  ``inspect``, and ``verify``. It records what a later agent should inspect
  without asserting an unverified fix.
- ``verified_repair``: contain ``when``, ``observed_failure``, ``avoid``, ``do``,
  and ``verify``. Use this only when the Analyzer marks the causal attribution
  direct and supplies a grounded repair hypothesis.

Rules:
1. Use only the supplied Analyzer failure evidence. A failed check grounds the
   observed failure pattern, but it does not prove an unsupported causal claim.
2. Remove task-specific constants, including task names, filenames, worksheet
   names, cell addresses, record IDs, exact answers, verifier function names,
   organization names, and one-off values.
3. Combine related failures into one lesson. Do not emit duplicate or overlapping
   lessons, including lessons already covered by ``current_family_memory``.
4. Do not emit generic advice, unsupported causes, complete task solutions, or
   successful procedures that belong in the family skill bank.
5. Preserve operational details, but never upgrade a causal hypothesis into a
   command. For diagnostic lessons, ``inspect`` must be a concrete investigation.
   For verified repairs, ``do`` must be executable rather than vague.
6. If two repairs interact or conflict, combine them into one coherent ordered
   lesson that resolves the tradeoff rather than emitting contradictory lessons.
7. Do not rely solely on an external tool unless the evidence establishes that
   it is available. Include an in-scope fallback or a validation step that
   detects when the proposed repair did not work.
8. If removing task-specific details leaves no concrete reusable repair lesson,
   return an empty ``lessons`` list.
9. For ``supporting_check_ids``, copy one or more strings verbatim from
   ``allowed_supporting_checks``. Never paraphrase, summarize, or invent an ID.
10. For ``evidence_refs``, copy one or more strings verbatim from
   ``allowed_evidence_refs``. Never construct a new reference.
11. If the failure directly contradicts an active skill claim, copy that exact
   claim ID into ``related_skill_claim_ids``. Otherwise return an empty list.
12. Remove one-off constants, but do not erase a recognizable exact family
   contract. Exact keys, locations, naming conventions, and serialization
   properties may be marked ``provisional_family_invariant``; use
   ``confirmed_family_invariant`` only when current memory shows support from a
   prior distinct task.
13. Return exactly one JSON object matching the requested schema, without Markdown
   or any text outside the JSON object.
"""
LESSON_FIELDS = (("when", 160), ("observed_failure", 180), ("avoid", 180), ("do", 180), ("inspect", 180), ("verify", 160))
CELL_RE = re.compile(r"\b[A-Z]{1,3}\$?\d+\b")


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesson_type: Literal["diagnostic", "verified_repair"] = "verified_repair"
    when: str = Field(min_length=1, max_length=160)
    observed_failure: str | None = Field(default=None, max_length=180)
    avoid: str | None = Field(default=None, max_length=180)
    do: str | None = Field(default=None, max_length=180)
    inspect: str | None = Field(default=None, max_length=180)
    verify: str = Field(min_length=1, max_length=160)
    supporting_check_ids: list[str] = Field(min_length=1, description="Exact, verbatim identifiers copied from allowed_supporting_checks.")
    evidence_refs: list[str] = Field(min_length=1, description="Exact, verbatim identifiers copied from allowed_evidence_refs.")
    related_skill_claim_ids: list[str] = Field(default_factory=list)
    invariant_scope: Literal["task_pattern", "provisional_family_invariant", "confirmed_family_invariant"] = "task_pattern"

    @model_validator(mode="after")
    def validate_lesson_contract(self) -> "MemoryCandidate":
        if self.lesson_type == "diagnostic":
            if not self.observed_failure or not self.inspect:
                raise ValueError("diagnostic lessons require observed_failure and inspect")
            if self.avoid is not None or self.do is not None:
                raise ValueError("diagnostic lessons must not contain avoid or do")
        elif not self.avoid or not self.do:
            raise ValueError("verified_repair lessons require avoid and do")
        return self


class MemoryCuratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lessons: list[MemoryCandidate] = Field(max_length=2)


@dataclass
class MemoryDecision:
    operation: str
    memory_id: str
    reason: str
    source_task: str
    supporting_checks: list[str]


@dataclass
class FamilyMemoryRun:
    report: dict[str, Any]
    decisions: list[MemoryDecision]
    token_usage: TokenUsage


def _load_state(memory_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(memory_path.with_suffix(".state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = None
    return state if isinstance(state, dict) and isinstance(state.get("items"), list) else {"items": []}


def _task_specific_problem(item: MemoryCandidate, task_name: str) -> str | None:
    fields = [item.when, item.observed_failure or "", item.avoid or "", item.do or "", item.inspect or "", item.verify]
    text = " ".join(fields).casefold()
    if len(task_name) >= 8 and task_name.casefold() in text:
        return "candidate contains the source task name"
    if any(len(check) >= 6 and check.casefold() in text for check in item.supporting_check_ids):
        return "candidate contains a verifier function name"
    if CELL_RE.search(" ".join(fields)) and item.invariant_scope == "task_pattern":
        return "candidate contains an exact cell-like contract but does not mark it as a provisional or confirmed family invariant"
    return None


def _validate_memory_response(response: MemoryCuratorResponse, failure_evidence: list[dict[str, Any]], task_name: str, allowed_claim_ids: list[str]) -> None:
    allowed_checks = {item["check"] for item in failure_evidence if isinstance(item.get("check"), str)}
    allowed_refs = {ref for item in failure_evidence for ref in item.get("evidence_refs", []) if isinstance(ref, str)}
    problems: list[str] = []
    for index, item in enumerate(response.lessons):
        invalid_checks = [check for check in item.supporting_check_ids if check not in allowed_checks]
        invalid_refs = [ref for ref in item.evidence_refs if ref not in allowed_refs]
        invalid_claim_ids = [claim_id for claim_id in item.related_skill_claim_ids if claim_id not in allowed_claim_ids]
        if invalid_checks:
            problems.append(f"lesson {index} has unsupported checks {invalid_checks!r}; copy exact values from allowed_supporting_checks: {sorted(allowed_checks)!r}")
        if invalid_refs:
            problems.append(f"lesson {index} has unsupported refs {invalid_refs!r}; copy exact values from allowed_evidence_refs: {sorted(allowed_refs)!r}")
        if invalid_claim_ids:
            problems.append(
                f"lesson {index} has unsupported related skill claims {invalid_claim_ids!r}; copy exact values from allowed_skill_claim_ids: {sorted(allowed_claim_ids)!r}"
            )
        if item.lesson_type == "verified_repair" and not any(
            evidence.get("check") in item.supporting_check_ids
            and evidence.get("causal_attribution") == "direct"
            and evidence.get("confidence") in {"high", "medium"}
            and isinstance(evidence.get("repair_hypothesis"), dict)
            for evidence in failure_evidence
        ):
            problems.append(
                f"lesson {index} is prescriptive but no supporting failure has direct causal attribution and a grounded repair hypothesis; emit a diagnostic lesson instead"
            )
        if specificity_problem := _task_specific_problem(item, task_name):
            problems.append(f"lesson {index}: {specificity_problem}")
    if problems:
        raise ValueError("; ".join(problems))


def _retain_grounded_lessons(
    response: MemoryCuratorResponse, failure_evidence: list[dict[str, Any]], task_name: str, allowed_claim_ids: list[str], allow_empty: bool = False
) -> MemoryCuratorResponse:
    valid: list[MemoryCandidate] = []
    for lesson in response.lessons:
        try:
            _validate_memory_response(response.model_copy(update={"lessons": [lesson]}), failure_evidence, task_name, allowed_claim_ids)
        except ValueError:
            continue
        valid.append(lesson)
    if not valid and response.lessons and not allow_empty:
        _validate_memory_response(response, failure_evidence, task_name, allowed_claim_ids)
    return response.model_copy(update={"lessons": valid})


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("lessons"), list):
        return payload
    normalized: list[dict[str, Any]] = []
    for raw_lesson in payload["lessons"][:2]:
        if not isinstance(raw_lesson, dict):
            continue
        lesson = dict(raw_lesson)
        for name, limit in LESSON_FIELDS:
            if isinstance(lesson.get(name), str):
                lesson[name] = re.sub(r"\s+", " ", lesson[name]).strip()[:limit].rstrip()
        text = " ".join(str(lesson.get(name) or "") for name, _ in LESSON_FIELDS)
        if lesson.get("invariant_scope") == "task_pattern" and CELL_RE.search(text):
            lesson["invariant_scope"] = "provisional_family_invariant"
        normalized.append(lesson)
    return {**payload, "lessons": normalized}


class FamilyRepairMemoryCurator:
    def __init__(
        self,
        *,
        model_name: str = "claude-opus-4.6",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        api_base: str | None = None,
        transport: str = "api",
        max_output_tokens: int = 4_096,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        client_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "api_key_env": api_key_env,
            "transport": transport,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if api_base:
            client_kwargs["api_base"] = api_base
        self.client = AzureOpenAILLM(**client_kwargs)

    def curate(
        self,
        *,
        family_name: str,
        task_name: str,
        task_position: int,
        analyzer_report: dict[str, Any],
        memory_path: Path,
        log_prefix: Path,
    ) -> tuple[MemoryCuratorResponse | None, dict[str, Any], TokenUsage]:
        failure_evidence = [item for item in analyzer_report.get("failure_evidence") or [] if isinstance(item, dict)]
        allowed_claim_ids = list(
            dict.fromkeys(
                str(item["claim_id"])
                for item in analyzer_report.get("skill_claim_feedback") or []
                if isinstance(item, dict) and item.get("outcome") == "contradicted" and item.get("claim_id")
            )
        )
        curator_input = {
            "family": family_name,
            "source_task": task_name,
            "task_position": task_position,
            "analysis_status": analyzer_report.get("analysis_status"),
            "failure_evidence": failure_evidence,
            "allowed_supporting_checks": list(dict.fromkeys(item["check"] for item in failure_evidence if isinstance(item.get("check"), str))),
            "allowed_evidence_refs": list(dict.fromkeys(ref for item in failure_evidence for ref in item.get("evidence_refs", []) if isinstance(ref, str))),
            "allowed_skill_claim_ids": allowed_claim_ids,
            "skill_claim_feedback": analyzer_report.get("skill_claim_feedback") or [],
            "current_family_memory": _load_state(memory_path),
        }
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        log_prefix.with_suffix(".request.json").write_text(json.dumps(curator_input, ensure_ascii=False, indent=2), encoding="utf-8")
        report: dict[str, Any] = {"curator_model": self.model_name, "lessons": []}
        if analyzer_report.get("analysis_status") != "complete" or not failure_evidence:
            report |= {"curation_status": "skipped", "summary": "No schema-valid Analyzer failure evidence to curate.", "curator_errors": []}
            return None, report, zero_usage()
        prompt = (
            "Return JSON matching this schema:\n"
            + json.dumps(MemoryCuratorResponse.model_json_schema(), ensure_ascii=False)
            + '\n\nThe required top-level format is: {"lessons":[{"lesson_type":"diagnostic","when":"...",'
            '"observed_failure":"...","avoid":null,"do":null,"inspect":"...",'
            '"verify":"...","supporting_check_ids":["..."],'
            '"evidence_refs":["..."],"related_skill_claim_ids":[],'
            '"invariant_scope":"task_pattern"}]}. Return {"lessons":[]} when no reusable grounded lesson exists.'
            "\n\nIMPORTANT: supporting_check_ids and evidence_refs are closed "
            "vocabularies. Copy their strings exactly from allowed_supporting_checks "
            "and allowed_evidence_refs in CURATOR INPUT. Do not paraphrase them.\n\nCURATOR INPUT:\n" + json.dumps(curator_input, ensure_ascii=False)
        )
        llm_run = call_structured_llm(
            client=self.client,
            model_type=MemoryCuratorResponse,
            system_prompt=MEMORY_CURATOR_SYSTEM_PROMPT,
            prompt=prompt,
            log_prefix=log_prefix,
            max_output_tokens=self.max_output_tokens,
            max_attempts=2,
            usage_source="azure_repair_memory_curator",
            model_name=self.model_name,
            payload_transform=_normalize_payload,
            value_transform=lambda value: _retain_grounded_lessons(value, failure_evidence, task_name, allowed_claim_ids),
            failure_value_transform=lambda value: _retain_grounded_lessons(value, failure_evidence, task_name, allowed_claim_ids, allow_empty=True),
            validator=lambda value: _validate_memory_response(value, failure_evidence, task_name, allowed_claim_ids),
        )
        response = llm_run.value
        if not isinstance(response, MemoryCuratorResponse):
            report |= {"curation_status": "failed", "summary": "Memory Curator produced no schema-valid grounded result.", "curator_errors": llm_run.errors}
            return None, report, llm_run.token_usage
        report |= {"curation_status": "complete", **response.model_dump(mode="json"), "curator_errors": llm_run.errors}
        return response, report, llm_run.token_usage


def _fingerprint(item: dict[str, Any]) -> str:
    fields = [
        item.get("lesson_type", ""),
        item.get("trigger", ""),
        item.get("observed_failure") or item.get("failure_pattern", ""),
        item.get("inspection_action", ""),
        item.get("corrective_action") or item.get("avoid", ""),
        item.get("verify", ""),
    ]
    return re.sub(r"\W+", " ", " ".join(str(value) for value in fields).casefold()).strip()


def _render_family_memory(items: list[dict[str, Any]]) -> str:
    active = [item for item in items if item.get("status") == "active"]
    lines = ["# Family Repair Memory", "", "## Relevant Prior Family Lessons"]
    if not active:
        return "\n".join([*lines, "", "- No active family repair lessons yet.", ""])
    for item in active:
        lines += ["", f"- When: {item['trigger']}"]
        if item["lesson_type"] == "diagnostic":
            observed = item["observed_failure"]
            lines += [
                f"  Observed: {observed}",
                "  Inspect: " + (item["inspection_action"] or f"Check current evidence to determine whether this recurs: {observed}"),
            ]
        else:
            lines += [f"  Avoid: {item['failure_pattern']}", f"  Do: {item['corrective_action'] or item['avoid']}"]
        lines.append(f"  Verify: {item['verify']}")
    return "\n".join([*lines, ""])


def _apply_lessons(memory_path: Path, task_name: str, task_position: int, response: MemoryCuratorResponse | None) -> list[MemoryDecision]:
    if response is None:
        return []
    state = _load_state(memory_path)
    items = state["items"]
    decisions: list[MemoryDecision] = []
    for candidate in response.lessons:
        checks = list(candidate.supporting_check_ids)
        if problem := _task_specific_problem(candidate, task_name):
            decisions.append(MemoryDecision("reject", "", problem, task_name, checks))
            continue
        observed_failure = candidate.observed_failure or candidate.avoid or ""
        data = {
            "lesson_type": candidate.lesson_type,
            "trigger": candidate.when,
            "observed_failure": observed_failure,
            "failure_pattern": observed_failure,
            "corrective_action": candidate.do,
            "inspection_action": candidate.inspect,
            "avoid": candidate.do or candidate.inspect or "",
            "verify": candidate.verify,
            "invariant_scope": candidate.invariant_scope,
        }
        target = next((item for item in items if item.get("status") == "active" and _fingerprint(item) == _fingerprint(data)), None)
        operation = "merge" if target else "add"
        if target is None:
            base_id = (re.sub(r"[^a-z0-9]+", "-", candidate.when.casefold()).strip("-")[:64]) or "repair-item"
            existing_ids = {item.get("id") for item in items}
            memory_id, suffix = base_id, 2
            while memory_id in existing_ids:
                memory_id, suffix = f"{base_id}-{suffix}", suffix + 1
            target = {"id": memory_id, "supporting_tasks": []}
            items.append(target)
        prior_scope = target.get("invariant_scope")
        for key in ("supporting_checks", "related_skill_claim_ids"):
            data[key] = list(
                dict.fromkeys([*target.get(key, []), *(candidate.supporting_check_ids if key == "supporting_checks" else candidate.related_skill_claim_ids)])
            )
        if prior_scope == "provisional_family_invariant" and task_name not in target["supporting_tasks"]:
            data["invariant_scope"] = "confirmed_family_invariant"
        target.update(data, status="active", last_task_position=task_position)
        if task_name not in target["supporting_tasks"]:
            target["supporting_tasks"].append(task_name)
        reason = "Merged exact duplicate lesson and accumulated verifier evidence." if operation == "merge" else "Added a schema-valid grounded family repair lesson."
        decisions.append(MemoryDecision(operation, target["id"], reason, task_name, checks))
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.with_suffix(".state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    memory_path.write_text(_render_family_memory(items), encoding="utf-8")
    return decisions


def curate_family_repair_memory(
    *,
    curator: FamilyRepairMemoryCurator,
    memory_path: Path,
    family_name: str,
    task_name: str,
    task_position: int,
    analyzer_report: dict[str, Any],
    log_prefix: Path,
) -> FamilyMemoryRun:
    response, report, usage = curator.curate(
        family_name=family_name,
        task_name=task_name,
        task_position=task_position,
        analyzer_report=analyzer_report,
        memory_path=memory_path,
        log_prefix=log_prefix,
    )
    decisions = _apply_lessons(memory_path, task_name, task_position, response)
    report["pipeline_schema_version"] = 2
    report["decisions"] = [asdict(decision) for decision in decisions]
    report["token_usage"] = token_usage_dict(usage)
    log_prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return FamilyMemoryRun(report=report, decisions=decisions, token_usage=usage)
