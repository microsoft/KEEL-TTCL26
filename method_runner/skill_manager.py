# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
import shutil
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from libs.azure_agents.azure_llm import AzureOpenAILLM
from method_runner.llm_json import call_structured_llm
from method_runner.token_usage import TokenUsage, token_usage_dict, zero_usage

FRONTMATTER_RE = re.compile(r"\A---\nname:\s*([A-Za-z0-9_.-]+)\ndescription:\s*(.+?)\n---\n", re.DOTALL)
NUMERIC_COMPUTATION_RE = re.compile(r"\b(sum(?!mar)|aggregate|rank|compute|calculate|classify)\w*\b", re.IGNORECASE)
TRANSFORMATION_STEP_RE = re.compile(r"\b(filter|convert|join|sort)\w*\b", re.IGNORECASE)
STRUCTURAL_OPERATION_CONTEXT_RE = re.compile(
    r"\b(formula|cell|range|sheet|worksheet|workbook|column|row|reference|defined name|named range|layout|header|table|pivot)\b", re.IGNORECASE
)
NEGATED_OPERATION_PREFIX_RE = re.compile(r"\b(?:without|never|avoid|do not|does not|must not)\b[^.;:]{0,80}$", re.IGNORECASE)
PARAMETER_VALUE_RES = [
    re.compile(r"""[="'`]|\b[A-Za-z]:[\\/]|(?:^|\s)/(?:[^/\s]+/)"""),
    re.compile(r"\b(?=[A-Za-z0-9_-]{12,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+\b"),
    re.compile(r"\b[A-Fa-f0-9]{16,}\b"),
]
ACTIVE_STATUSES = {"provisional", "confirmed"}
ROUTING_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")
ROUTING_STOPWORDS = {"and", "are", "for", "from", "into", "not", "only", "output", "task", "that", "the", "their", "then", "this", "use", "when", "with"}
MAX_ATTEMPTS = 2

SKILL_CURATOR_SYSTEM_PROMPT = """You are the Positive Skill Curator in a verifier-guided lifelong-learning experiment. You do not operate a terminal.
Convert grounded positive behavior into reusable SKILL.md files for later tasks in the same workflow family when safe abstraction leaves an actionable procedure.

Rules:
1. Use only verifier-supported positive evidence supplied in the input. The Analyzer's family_generalizable field is advisory: you may extract a family skill
   from an item marked false when removing its task-specific constants leaves a grounded, actionable procedure.
2. Do not create a skill from failure evidence or an unverified repair hypothesis.
3. Remove task-specific constants and create a reusable procedure when the verified behavior can be safely abstracted. Reject only when removing one-off
   constants leaves no actionable procedure. Always remove source task names, organization-specific identifiers, exact answers, verifier function names, and
   one-off values. Exact keys, cell locations, naming conventions, or serialization properties may remain only when the supplied positive evidence directly
   supports them as a recognizable family contract. Otherwise use abstract parameter labels such as "required output path" or "requested reporting period"
   under task_resolved_parameters. Never copy the source task's resolved value into a reusable skill.
4. A skill must state when to use it, an actionable procedure, and verification. Its one-line description is a retrieval contract: begin with "Use when",
   identify the artifact type and operation or situation, and include concrete trigger terms that a later task instruction is likely to contain. Avoid vague
   descriptions such as "helps with this family" or "useful for data tasks".
5. Inspect existing skills. Do not emit a redundant candidate. To improve an existing skill, reuse its name and return a complete consolidated body.
6. Return each candidate as a name, description, structured procedure and verification steps, claim records, and grounded applicability metadata. Do not
   return a free-form Markdown body. Every step must declare its step_scope and cite one or more candidate claim IDs. Procedure steps use the same scope as
   their supporting claims. A verification step may use step_scope "verification" while citing the underlying claims it verifies. Every candidate claim must
   cite exact Analyzer behavior-claim IDs, passed checks, trajectory steps, and evidence references. Use short normalized phrases for artifact_types and
   operations, task-instruction cues for triggers, and exclusions only when the supplied evidence establishes a useful boundary. Do not return an
   add/merge/reject operation, path, or YAML frontmatter.
7. Reject weakly grounded or redundant behavior. Do not reject merely because the evidence contains task-specific constants when an actionable verified
   procedure remains after removing them.
8. Return exactly one JSON object matching the requested schema, without Markdown.
9. A file-existence claim can support only creating/writing the required path. It cannot support filtering, computation, ranking, transformation, or other
   semantic procedures. Apply the same scope discipline to structure and formatting checks.
10. A semantic-computation claim sourced from direct procedural attribution may become active according to lifecycle rules. A claim sourced only from
    compatible attribution may still be proposed: deterministic Python stores it as a non-mounted draft, and a later independent task must support the same
    claim before it can become provisional. Never propose a semantic claim from insufficient procedural attribution.
11. Keep every procedure step within one claim scope. Filtering, converting, joining, and sorting are transformations and require only transformation or
    semantic-computation claims. Do not combine one of those operations with a structure-only write/serialization action in the same step. When evidence
    supports only a structural order property, verify that order but do not invent a transformation procedure that the evidence does not support.
"""
ClaimScope = Literal["existence", "structure", "formatting", "transformation", "semantic_computation", "preservation", "verification"]
SCOPE_SAFE_PROCEDURE_INSTRUCTIONS = {
    "existence": "Create the required output artifact at the task-resolved path.",
    "structure": "Construct the task-resolved output containers and fields according to the grounded structural requirements.",
    "formatting": "Apply the grounded value types, ordering, and serialization formatting to the task-resolved output.",
    "preservation": "Preserve the grounded task-required properties while producing the output artifact.",
}


@dataclass
class SkillDecision:
    operation: str
    skill_name: str
    path: str
    reason: str
    support_checks: list[str]
    source_task: str
    iteration: int
    claim_id: str | None = None
    previous_status: str | None = None
    new_status: str | None = None
    evidence_refs: list[str] | None = None


@dataclass
class SkillCuratorRun:
    report: dict[str, Any]
    decisions: list[SkillDecision]
    token_usage: TokenUsage


def append_skill_decisions(path: Path, decisions: list[SkillDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _dicts(items: Any) -> list[dict[str, Any]]:
    return [item for item in items or [] if isinstance(item, dict)]


def _bad_parameter(value: str) -> bool:
    return any(pattern.search(value) for pattern in PARAMETER_VALUE_RES)


class SkillApplicability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_types: list[str] = Field(min_length=1)
    operations: list[str] = Field(min_length=1)
    triggers: list[str] = Field(min_length=1)
    exclusions: list[str] = Field(default_factory=list)

    @field_validator("artifact_types", "operations", "triggers", "exclusions")
    @classmethod
    def phrases_must_be_short_and_unique(cls, values: list[str]) -> list[str]:
        phrases = {value.strip().casefold(): value.strip() for value in values}
        if any(not phrase or "\n" in phrase or len(phrase) > 160 for phrase in phrases.values()):
            raise ValueError("applicability phrases must be non-blank single lines of at most 160 characters")
        return list(phrases.values())


class SkillStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(pattern=r"^(?:procedure|verification)\.[A-Za-z0-9_.-]+$")
    instruction: str = Field(min_length=1, max_length=600)
    supporting_claim_ids: list[str] = Field(min_length=1)
    step_scope: ClaimScope | None = None


class SkillClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.claim\.[A-Za-z0-9_.-]+$")
    claim: str = Field(min_length=1, max_length=600)
    claim_scope: ClaimScope
    source_behavior_claim_ids: list[str] = Field(min_length=1)
    supporting_checks: list[str] = Field(min_length=1)
    trajectory_step_refs: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class SkillCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=160)
    when_to_use: str = Field(min_length=1, max_length=600)
    procedure_steps: list[SkillStep] = Field(min_length=1)
    verification_steps: list[SkillStep] = Field(min_length=1)
    claims: list[SkillClaim] = Field(min_length=1)
    task_resolved_parameters: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    applicability: SkillApplicability
    supporting_checks: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def description_must_be_one_line(cls, value: str) -> str:
        value = value.strip()
        if "\n" in value or len(value) > 320 or not value.casefold().startswith("use when "):
            raise ValueError('description must be one line of at most 320 characters beginning with "Use when"')
        return value

    @field_validator("task_resolved_parameters")
    @classmethod
    def parameters_must_be_labels_not_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or _bad_parameter(value.strip()) for value in values):
            raise ValueError("task_resolved_parameters must contain abstract labels, not assignments, quoted values, absolute paths, or task identifiers")
        return values


class SkillCuratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    candidate_skills: list[SkillCandidate]


def _contains_numeric_computation(instruction: str) -> bool:
    return any(not NEGATED_OPERATION_PREFIX_RE.search(instruction[max(0, match.start() - 100) : match.start()]) for match in NUMERIC_COMPUTATION_RE.finditer(instruction))


def _scope_violation(instruction: str, scope: str | None) -> bool:
    if scope in {"transformation", "semantic_computation"}:
        return False
    if scope == "structure" and STRUCTURAL_OPERATION_CONTEXT_RE.search(instruction):
        return False
    return _contains_numeric_computation(instruction) or bool(TRANSFORMATION_STEP_RE.search(instruction))


def _normalize_procedure_instruction_scope(instruction: str, scope: str) -> str:
    return SCOPE_SAFE_PROCEDURE_INSTRUCTIONS.get(scope, instruction) if _scope_violation(instruction, scope) else instruction


def _active_claim_ids(metadata: dict[str, Any], suppressed: set[str] = frozenset()) -> set[str]:
    return {str(item.get("claim_id")) for item in _dicts(metadata.get("claims")) if item.get("status") in ACTIVE_STATUSES and item.get("claim_id") not in suppressed}


def _lifecycle_status(claims: list[dict[str, Any]], fallback: str) -> str:
    active = {item.get("status") for item in claims if item.get("status") in ACTIVE_STATUSES}
    return "confirmed" if active == {"confirmed"} else "provisional" if active else fallback


def render_skill_markdown(candidate: SkillCandidate, *, active_claim_ids: set[str] | None = None) -> str:
    active = {claim.claim_id for claim in candidate.claims} if active_claim_ids is None else active_claim_ids
    procedure = [step for step in candidate.procedure_steps if all(claim_id in active for claim_id in step.supporting_claim_ids)]
    verification = [step for step in candidate.verification_steps if all(claim_id in active for claim_id in step.supporting_claim_ids)]
    lines = [f"# {candidate.title}", "", "## When to use", "", candidate.when_to_use.strip(), "", "## Procedure", ""]
    lines += [f"{index}. {step.instruction.strip()}" for index, step in enumerate(procedure, start=1)]
    lines += ["", "## Verification", ""] + [f"- {step.instruction.strip()}" for step in verification]
    if candidate.limitations:
        lines += ["", "## Limitations", ""] + [f"- {item.strip()}" for item in candidate.limitations]
    body = "\n".join(lines).strip()
    return f"---\nname: {candidate.name}\ndescription: {candidate.description}\n---\n\n{body}\n"


def render_applicability_metadata(
    candidate: SkillCandidate,
    *,
    source_task: str,
    existing_metadata: dict[str, Any] | None,
    initial_status_by_claim: dict[str, str],
) -> dict[str, Any]:
    prior_by_id = {item["claim_id"]: item for item in _dicts((existing_metadata or {}).get("claims")) if item.get("claim_id")}
    candidate_ids = {claim.claim_id for claim in candidate.claims}
    claims: list[dict[str, Any]] = []
    for claim in candidate.claims:
        prior = prior_by_id.get(claim.claim_id) or {}
        status = str(prior.get("status") or initial_status_by_claim.get(claim.claim_id) or "provisional")
        if status not in {"draft", "provisional", "confirmed", "contradicted", "retired"}:
            status = "provisional"
        supporting_tasks = list(dict.fromkeys(list(prior.get("supporting_tasks") or []) + [source_task]))
        if status == "draft" and len(supporting_tasks) >= 2 and str(prior.get("introduced_task") or source_task) != source_task:
            status = "provisional"
        claims.append(
            {
                **claim.model_dump(mode="json"),
                "status": status,
                "introduced_task": prior.get("introduced_task") or source_task,
                "supporting_tasks": supporting_tasks,
                "contradiction_tasks": list(prior.get("contradiction_tasks") or []),
            }
        )
    for claim_id, prior in prior_by_id.items():
        if claim_id not in candidate_ids:
            claims.append({**prior, "status": "retired", "retired_task": source_task, "retirement_reason": "Superseded by a complete grounded replacement."})
    current = [item for item in claims if item["claim_id"] in candidate_ids]
    fallback = "draft" if any(item["status"] == "draft" for item in current) else "retired"
    return {
        "name": candidate.name,
        "description": candidate.description,
        "lifecycle_status": _lifecycle_status(current, fallback),
        "applicability": candidate.applicability.model_dump(mode="json"),
        "supporting_checks": candidate.supporting_checks,
        "evidence_refs": candidate.evidence_refs,
        "claims": claims,
        "candidate": candidate.model_dump(mode="json"),
        "task_resolved_parameters": candidate.task_resolved_parameters,
        "limitations": candidate.limitations,
        "conflict_history": list((existing_metadata or {}).get("conflict_history") or []),
    }


def _eligible_positive_evidence(analyzer_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in analyzer_report.get("positive_behavior_evidence") or []
        if item.get("status") == "verifier_supported_positive" and item.get("support") == "1/1" and item.get("confidence") in {"high", "medium"}
    ]


def _existing_skill_input(skills_dir: Path, max_chars: int = 60_000) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    remaining = max_chars
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        metadata = _load_json(path.parent / "applicability.json")
        if not metadata or metadata.get("lifecycle_status") in {"contradicted", "retired"}:
            continue
        active = _active_claim_ids(metadata)
        has_draft = any(item.get("status") == "draft" for item in _dicts(metadata.get("claims")))
        if not active and not has_draft:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        try:
            content = render_skill_markdown(SkillCandidate.model_validate(metadata["candidate"]), active_claim_ids=None if has_draft else active)
        except (KeyError, ValueError):
            pass
        if remaining <= 0:
            break
        content = content[:remaining]
        remaining -= len(content)
        existing.append(
            {
                "name": path.parent.name,
                "description": metadata.get("description", ""),
                "path": path.relative_to(skills_dir).as_posix(),
                "content": content,
                "applicability_metadata": metadata,
            }
        )
    return existing


def _specificity_problem(content: str, *, task_name: str, checks: list[str], candidate: SkillCandidate) -> str | None:
    routing = json.dumps(
        {"name": candidate.name, "description": candidate.description, "applicability": candidate.applicability.model_dump(mode="json")}, ensure_ascii=False
    )
    folded = f"{FRONTMATTER_RE.sub('', content, count=1)}\n{routing}".casefold()
    if len(task_name) >= 8 and task_name.casefold() in folded:
        return "Candidate contains the source task name."
    if any(len(check) >= 6 and check.casefold() in folded for check in checks):
        return "Candidate contains a verifier function name."
    return None


def _skill_content_problem(candidate: SkillCandidate, content: str, task_name: str) -> str | None:
    match = FRONTMATTER_RE.match(content)
    if not match or match.group(1) != candidate.name:
        return "missing required name/description frontmatter"
    return _specificity_problem(content, task_name=task_name, checks=candidate.supporting_checks, candidate=candidate)


def _validate_skill_curator_response(response: SkillCuratorResponse, eligible: list[dict[str, Any]], *, task_name: str) -> None:
    behavior_claims = {claim["claim_id"]: claim for item in eligible for claim in _dicts(item.get("behavior_claims")) if isinstance(claim.get("claim_id"), str)}
    problems: list[str] = []
    for index, candidate in enumerate(response.candidate_skills):
        prefix = f"candidate {index}"
        claim_ids = [claim.claim_id for claim in candidate.claims]
        if len(claim_ids) != len(set(claim_ids)):
            problems.append(f"{prefix} has duplicate claim IDs")
        for claim in candidate.claims:
            label = f"{prefix} claim {claim.claim_id!r}"
            if not claim.claim_id.startswith(f"{candidate.name}.claim."):
                problems.append(f"{label} must begin with {candidate.name}.claim.")
            sources = [behavior_claims[source] for source in claim.source_behavior_claim_ids if source in behavior_claims]
            if len(sources) != len(claim.source_behavior_claim_ids):
                problems.append(f"{label} has unsupported source behavior claims")
            source_scopes = {source.get("claim_scope") for source in sources}
            if source_scopes and claim.claim_scope not in source_scopes:
                problems.append(f"{label} widens source scope {sorted(str(scope) for scope in source_scopes)!r} to {claim.claim_scope!r}")
            if claim.claim_scope == "semantic_computation" and any(source.get("procedural_attribution") == "insufficient" for source in sources):
                problems.append(f"{label} is semantic but uses insufficient procedural attribution")
        claim_id_set = set(claim_ids)
        steps = candidate.procedure_steps + candidate.verification_steps
        for step in steps:
            label = f"{prefix} step {step.step_id!r}"
            if not set(step.supporting_claim_ids) <= claim_id_set:
                problems.append(f"{label} has unsupported claim IDs")
            supporting_scopes = {claim.claim_scope for claim in candidate.claims if claim.claim_id in step.supporting_claim_ids}
            if step.step_scope is None:
                problems.append(f"{label} has no step_scope")
            elif not (step.step_id.startswith("verification.") and step.step_scope == "verification") and supporting_scopes != {step.step_scope}:
                problems.append(f"{label} scope {step.step_scope!r} does not exactly match supporting claim scopes {sorted(supporting_scopes)!r}")
            if step.step_id.startswith("procedure.") and _scope_violation(step.instruction, step.step_scope):
                problems.append(f"{label} contains computation or data transformation but is not scoped transformation or semantic_computation")
        if claim_id_set - {claim_id for step in steps for claim_id in step.supporting_claim_ids}:
            problems.append(f"{prefix} has claims not linked to a rendered step")
        if len({step.step_id for step in steps}) != len(steps):
            problems.append(f"{prefix} has duplicate step IDs")
        problem = _skill_content_problem(candidate, render_skill_markdown(candidate), task_name)
        if problem:
            problems.append(f"{prefix}: {problem}")
    if problems:
        raise ValueError("; ".join(problems))


def _one_line(value: Any, limit: int) -> Any:
    return re.sub(r"\s+", " ", value).strip()[:limit].rstrip() if isinstance(value, str) else value


def _normalize_candidate(raw: dict[str, Any], allowed_checks: set, allowed_refs: set, allowed_steps: set, source_scopes: dict) -> dict[str, Any]:
    candidate = dict(raw)
    name = candidate.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return candidate
    description = _one_line(candidate.get("description"), 311)
    if isinstance(description, str) and description and not description.casefold().startswith("use when "):
        description = f"Use when {description}"[:320].rstrip()
    candidate["description"] = description
    candidate["title"] = _one_line(candidate.get("title"), 160)
    candidate["when_to_use"] = _one_line(candidate.get("when_to_use"), 600)
    if isinstance(candidate.get("task_resolved_parameters"), list):
        candidate["task_resolved_parameters"] = [
            value.strip() for value in candidate["task_resolved_parameters"] if isinstance(value, str) and value.strip() and not _bad_parameter(value)
        ]
    if isinstance(candidate.get("applicability"), dict):
        applicability = dict(candidate["applicability"])
        for field in ("artifact_types", "operations", "triggers", "exclusions"):
            if isinstance(applicability.get(field), list):
                applicability[field] = [_one_line(value, 160) for value in applicability[field] if isinstance(value, str)]
        candidate["applicability"] = applicability
    if not isinstance(candidate.get("claims"), list):
        return candidate
    claims: list[dict[str, Any]] = []
    claim_id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for index, raw_claim in enumerate(_dicts(candidate["claims"]), start=1):
        claim = dict(raw_claim)
        old_id = str(claim.get("claim_id") or f"claim-{index}")
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", old_id.split(".claim.", 1)[-1]).strip("-.") or str(index)
        new_id = f"{name}.claim.{suffix}"
        serial = 2
        while new_id in used_ids:
            new_id = f"{name}.claim.{suffix}-{serial}"
            serial += 1
        used_ids.add(new_id)
        claim_id_map[old_id] = new_id
        claim["claim_id"] = new_id
        claim["claim"] = _one_line(claim.get("claim"), 600)
        sources = [value for value in claim.get("source_behavior_claim_ids", []) if value in source_scopes]
        if not sources:
            continue
        matching = [value for value in sources if source_scopes[value] == claim.get("claim_scope")]
        if matching:
            sources = matching
        else:
            scopes = {source_scopes[value] for value in sources}
            if len(scopes) != 1:
                continue
            claim["claim_scope"] = scopes.pop()
        claim["source_behavior_claim_ids"] = sources
        claim["supporting_checks"] = [value for value in claim.get("supporting_checks", []) if value in allowed_checks]
        claim["trajectory_step_refs"] = [value for value in claim.get("trajectory_step_refs", []) if value in allowed_steps]
        claim["evidence_refs"] = [value for value in claim.get("evidence_refs", []) if value in allowed_refs]
        if all(claim[field] for field in ("supporting_checks", "trajectory_step_refs", "evidence_refs")):
            claims.append(claim)
    scopes_by_claim = {claim["claim_id"]: claim.get("claim_scope") for claim in claims}

    def normalize_steps(values: Any, *, verification: bool) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw_step in _dicts(values):
            step = dict(raw_step)
            step["instruction"] = _one_line(step.get("instruction"), 600)
            supporting = [claim_id_map.get(str(value), str(value)) for value in step.get("supporting_claim_ids", [])]
            supporting = [value for value in supporting if value in scopes_by_claim]
            if not supporting:
                continue
            scopes = {scopes_by_claim[value] for value in supporting}
            if verification:
                step["step_scope"] = "verification"
            else:
                instruction = step["instruction"] if isinstance(step["instruction"], str) else ""
                if len(scopes) == 1:
                    scope = next(iter(scopes))
                elif step.get("step_scope") in scopes:
                    scope = step["step_scope"]
                else:
                    continue
                supporting = [value for value in supporting if scopes_by_claim[value] == scope]
                step["step_scope"] = scope
                if instruction:
                    step["instruction"] = _normalize_procedure_instruction_scope(instruction, scope)
            step["supporting_claim_ids"] = supporting
            result.append(step)
        return result

    procedures = normalize_steps(candidate.get("procedure_steps"), verification=False)
    verifications = normalize_steps(candidate.get("verification_steps"), verification=True)
    linked = {claim_id for step in procedures + verifications for claim_id in step["supporting_claim_ids"]}
    candidate["claims"] = [claim for claim in claims if claim["claim_id"] in linked]
    candidate["procedure_steps"] = procedures
    candidate["verification_steps"] = verifications
    candidate["supporting_checks"] = [value for value in candidate.get("supporting_checks", []) if value in allowed_checks]
    candidate["evidence_refs"] = [value for value in candidate.get("evidence_refs", []) if value in allowed_refs]
    return candidate


def _normalize_skill_curator_payload(payload: dict[str, Any], eligible: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload.get("candidate_skills"), list):
        return payload
    allowed_checks = {item["check"] for item in eligible if isinstance(item.get("check"), str)}
    allowed_refs = {ref for item in eligible for ref in item.get("evidence_refs", []) if isinstance(ref, str)}
    allowed_steps = {ref for item in eligible for ref in item.get("trajectory_step_refs", []) if isinstance(ref, str)}
    source_scopes = {
        claim["claim_id"]: claim.get("claim_scope") for item in eligible for claim in _dicts(item.get("behavior_claims")) if isinstance(claim.get("claim_id"), str)
    }
    candidates = [_normalize_candidate(raw, allowed_checks, allowed_refs, allowed_steps, source_scopes) for raw in _dicts(payload["candidate_skills"])]
    return {**payload, "candidate_skills": candidates}


def _retain_grounded_skill_candidates(
    response: SkillCuratorResponse, eligible: list[dict[str, Any]], *, task_name: str, allow_empty: bool = False
) -> SkillCuratorResponse:
    valid: list[SkillCandidate] = []
    errors: list[str] = []
    for candidate in response.candidate_skills:
        try:
            _validate_skill_curator_response(response.model_copy(update={"candidate_skills": [candidate]}), eligible, task_name=task_name)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            valid.append(candidate)
    if not valid and response.candidate_skills and not allow_empty:
        raise ValueError("; ".join(errors))
    return response.model_copy(update={"candidate_skills": valid})


class LLMPositiveSkillCurator:
    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str,
        api_base: str | None = None,
        transport: str = "api",
        max_output_tokens: int = 6_144,
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
        skills_dir: Path,
        family_name: str,
        task_name: str,
        task_position: int,
        analyzer_report: dict[str, Any],
        log_prefix: Path,
    ) -> tuple[SkillCuratorResponse | None, dict[str, Any], TokenUsage]:
        eligible = _eligible_positive_evidence(analyzer_report)
        curator_input = {
            "family": family_name,
            "source_task": task_name,
            "task_position": task_position,
            "eligible_positive_evidence": eligible,
            "existing_skills": _existing_skill_input(skills_dir),
        }
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        log_prefix.with_suffix(".request.json").write_text(json.dumps(curator_input, ensure_ascii=False, indent=2), encoding="utf-8")
        base_report = {"curator_model": self.model_name, "candidate_skills": []}
        if analyzer_report.get("analysis_status") != "complete" or not eligible:
            summary = "No high- or medium-confidence verifier-supported positive evidence."
            return None, {**base_report, "curation_status": "skipped", "summary": summary, "curator_errors": []}, zero_usage()
        schema = SkillCuratorResponse.model_json_schema()
        prompt = "Return JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False) + "\n\nCURATOR INPUT:\n" + json.dumps(curator_input, ensure_ascii=False)
        llm_run = call_structured_llm(
            client=self.client,
            model_type=SkillCuratorResponse,
            system_prompt=SKILL_CURATOR_SYSTEM_PROMPT,
            prompt=prompt,
            log_prefix=log_prefix,
            max_output_tokens=self.max_output_tokens,
            max_attempts=MAX_ATTEMPTS,
            usage_source="azure_positive_skill_curator",
            model_name=self.model_name,
            payload_transform=lambda payload: _normalize_skill_curator_payload(payload, eligible),
            value_transform=lambda value: _retain_grounded_skill_candidates(value, eligible, task_name=task_name),
            failure_value_transform=lambda value: _retain_grounded_skill_candidates(value, eligible, task_name=task_name, allow_empty=True),
            validator=lambda value: _validate_skill_curator_response(value, eligible, task_name=task_name),
        )
        response = llm_run.value
        if not isinstance(response, SkillCuratorResponse):
            summary = "Skill Curator produced no schema-valid grounded result."
            return None, {**base_report, "curation_status": "failed", "summary": summary, "curator_errors": llm_run.errors}, llm_run.token_usage
        report = {"curation_status": "complete", "curator_model": self.model_name, **response.model_dump(mode="json"), "curator_errors": llm_run.errors}
        return response, report, llm_run.token_usage


def _initial_claim_statuses(candidate: SkillCandidate, analyzer_report: dict[str, Any] | None) -> dict[str, str]:
    if not analyzer_report:
        return {}
    partition = analyzer_report.get("partition") or {}
    full_success = bool(partition.get("passed")) and not any(partition.get(key) for key in ("failed", "unresolved", "infrastructure_error"))
    attribution = {
        str(claim["claim_id"]): str(claim.get("procedural_attribution") or "compatible")
        for item in _dicts(analyzer_report.get("positive_behavior_evidence"))
        for claim in _dicts(item.get("behavior_claims"))
        if claim.get("claim_id")
    }
    statuses: dict[str, str] = {}
    for claim in candidate.claims:
        direct = {attribution.get(source, "compatible") for source in claim.source_behavior_claim_ids} == {"direct"}
        statuses[claim.claim_id] = "draft" if claim.claim_scope == "semantic_computation" and not (full_success and direct) else "provisional"
    return statuses


def _normalized_skill(content: str) -> str:
    return re.sub(r"\W+", " ", content.casefold()).strip()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SkillBankManager:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir

    def apply_candidates(
        self,
        response: SkillCuratorResponse | None,
        *,
        task_name: str,
        iteration: int,
        analyzer_report: dict[str, Any] | None = None,
    ) -> list[SkillDecision]:
        decisions: list[SkillDecision] = []
        for candidate in response.candidate_skills if response else []:
            rel_path = f"{candidate.name}/SKILL.md"
            target = self.skills_dir / rel_path
            sidecar = target.parent / "applicability.json"
            existing_metadata = _load_json(sidecar)
            checks = candidate.supporting_checks

            def decide(operation: str, reason: str) -> None:
                decisions.append(SkillDecision(operation, candidate.name, rel_path, reason, checks, task_name, iteration))

            content = render_skill_markdown(candidate)
            problem = _skill_content_problem(candidate, content, task_name)
            if problem:
                decide("reject", problem)
                continue
            prior_claims = {str(item["claim_id"]): item for item in _dicts((existing_metadata or {}).get("claims")) if item.get("claim_id")}
            reused = sorted(claim.claim_id for claim in candidate.claims if prior_claims.get(claim.claim_id, {}).get("status") in {"contradicted", "retired"})
            if reused:
                decide("reject", "A contradicted or retired claim ID cannot be reactivated; emit a newly grounded corrected claim ID instead: " + ", ".join(reused))
                continue
            payload = render_applicability_metadata(
                candidate,
                source_task=task_name,
                existing_metadata=existing_metadata,
                initial_status_by_claim=_initial_claim_statuses(candidate, analyzer_report),
            )
            normalized = _normalized_skill(content)
            duplicate = next(
                (
                    path.relative_to(self.skills_dir).as_posix()
                    for path in sorted(self.skills_dir.glob("*/SKILL.md"))
                    if path != target and SequenceMatcher(None, normalized, _normalized_skill(path.read_text(encoding="utf-8"))).ratio() >= 0.94
                ),
                None,
            )
            if duplicate:
                decide("reject", f"Near-duplicate of existing {duplicate}.")
                continue
            operation = "merge" if target.is_file() else "add_draft" if payload["lifecycle_status"] == "draft" else "add_provisional"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            _write_json(sidecar, payload)
            decide(operation, candidate.rationale)
            for claim in payload["claims"]:
                previous = prior_claims.get(str(claim["claim_id"]))
                if claim["status"] == "retired" and previous and previous.get("status") != "retired":
                    decisions.append(
                        SkillDecision(
                            "retire",
                            candidate.name,
                            f"{candidate.name}/applicability.json",
                            str(claim.get("retirement_reason")),
                            list(claim.get("supporting_checks") or []),
                            task_name,
                            iteration,
                            claim["claim_id"],
                            str(previous.get("status") or "provisional"),
                            "retired",
                            list(claim.get("evidence_refs") or []),
                        )
                    )
        return decisions

    def apply_claim_feedback(self, analyzer_report: dict[str, Any], *, task_name: str, iteration: int) -> list[SkillDecision]:
        decisions: list[SkillDecision] = []
        for feedback in _dicts(analyzer_report.get("skill_claim_feedback")):
            skill_name = str(feedback.get("skill_name") or "")
            claim_id = str(feedback.get("claim_id") or "")
            sidecar = self.skills_dir / skill_name / "applicability.json"
            metadata = _load_json(sidecar) if skill_name and claim_id else None
            claim = next((item for item in _dicts((metadata or {}).get("claims")) if item.get("claim_id") == claim_id), None)
            if claim is None:
                continue
            previous = str(claim.get("status") or "provisional")
            outcome = feedback.get("outcome")
            checks = list(feedback.get("checks") or [])
            if outcome == "supported" and previous == "provisional" and claim.get("introduced_task") != task_name and checks:
                new_status, operation, task_list = "confirmed", "confirm", "supporting_tasks"
            elif (
                outcome == "contradicted"
                and previous in ACTIVE_STATUSES
                and feedback.get("confidence") in {"high", "medium"}
                and checks
                and feedback.get("trajectory_step_refs")
            ):
                new_status, operation, task_list = "contradicted", "contradict", "contradiction_tasks"
            else:
                continue
            tasks = claim.setdefault(task_list, [])
            if task_name not in tasks:
                tasks.append(task_name)
            claim["status"] = new_status
            reason = str(feedback.get("explanation") or "")
            evidence_refs = list(feedback.get("evidence_refs") or [])
            metadata.setdefault("conflict_history", []).append(
                {
                    "task": task_name,
                    "iteration": iteration,
                    "claim_id": claim_id,
                    "previous_status": previous,
                    "new_status": new_status,
                    "checks": checks,
                    "evidence_refs": evidence_refs,
                    "explanation": reason,
                }
            )
            metadata["lifecycle_status"] = _lifecycle_status(metadata["claims"], new_status)
            _write_json(sidecar, metadata)
            decisions.append(
                SkillDecision(
                    operation, skill_name, f"{skill_name}/applicability.json", reason, checks, task_name, iteration, claim_id, previous, new_status, evidence_refs
                )
            )
        return decisions


def _routing_keywords(value: Any) -> set[str]:
    if isinstance(value, dict):
        value = list(value.values())
    text = " ".join(str(item) for item in value) if isinstance(value, (list, tuple, set)) else str(value or "")
    return {word.casefold() for word in ROUTING_WORD_RE.findall(text) if word.casefold() not in ROUTING_STOPWORDS}


def _skill_routing_record(skill_name: str, metadata: dict[str, Any], task_keywords: set[str]) -> dict[str, Any]:
    candidate = metadata.get("candidate") or {}
    applicability = candidate.get("applicability") or metadata.get("applicability") or {}
    fields = {
        "description": candidate.get("description", ""),
        "when_to_use": candidate.get("when_to_use", ""),
        "artifact_types": applicability.get("artifact_types", []),
        "operations": applicability.get("operations", []),
        "triggers": applicability.get("triggers", []),
    }
    overlaps = {field: sorted(task_keywords & _routing_keywords(value)) for field, value in fields.items()}
    exclusion_matches: set[str] = set()
    for exclusion in applicability.get("exclusions", []):
        keywords = _routing_keywords(exclusion)
        overlap = task_keywords & keywords
        required = 1 if len(keywords) == 1 else max(2, (3 * len(keywords) + 4) // 5)
        if keywords and len(overlap) >= required:
            exclusion_matches.update(overlap)
    overlaps["exclusions"] = sorted(exclusion_matches)
    weights = {"description": 2, "when_to_use": 2, "artifact_types": 2, "operations": 2, "triggers": 3}
    return {
        "skill_name": skill_name,
        "score": sum(weight * len(overlaps[field]) for field, weight in weights.items()),
        "description": str(fields["description"]).strip(),
        "matched_terms": sorted({term for values in overlaps.values() for term in values}),
        "matched_fields": [field for field, values in overlaps.items() if values],
        "exclusion_matches": overlaps["exclusions"],
    }


def materialize_active_skill_snapshot(
    skills_dir: Path,
    snapshot_dir: Path,
    *,
    suppressed_claim_ids: set[str] | None = None,
    task_instruction: str | None = None,
    top_k: int | None = None,
    mount_selected_only: bool = False,
) -> dict[str, Any]:
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    suppressed = suppressed_claim_ids or set()
    candidates: list[tuple[str, dict[str, Any], set[str]]] = []
    suppressed_skills: list[str] = []
    suppressed_claim_count = 0
    for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
        skill_name = skill_path.parent.name
        metadata = _load_json(skill_path.parent / "applicability.json")
        if skill_name.startswith(".") or metadata is None:
            continue
        active = _active_claim_ids(metadata, suppressed)
        suppressed_claim_count += len(_active_claim_ids(metadata)) - len(active)
        if active:
            candidates.append((skill_name, metadata, active))
        else:
            suppressed_skills.append(skill_name)
    routing_records: list[dict[str, Any]] = []
    recommended: list[str] = []
    routed = task_instruction is not None and top_k is not None
    if routed:
        task_keywords = _routing_keywords(task_instruction)
        routing_records = sorted(
            (_skill_routing_record(name, metadata, task_keywords) for name, metadata, _ in candidates), key=lambda item: (-item["score"], item["skill_name"])
        )
        positive = [
            item
            for item in routing_records
            if item["score"] >= 8 and not item["exclusion_matches"] and {"artifact_types", "operations", "triggers"} & set(item["matched_fields"])
        ]
        if positive and positive[0]["score"] - (positive[1]["score"] if len(positive) > 1 else -1) >= 3:
            recommended = [item["skill_name"] for item in positive[: max(0, top_k)]]
    available: list[str] = []
    active_claim_count = 0
    for skill_name, metadata, active in candidates:
        if mount_selected_only and skill_name not in recommended:
            continue
        destination = snapshot_dir / skill_name
        destination.mkdir(parents=True, exist_ok=True)
        try:
            content = render_skill_markdown(SkillCandidate.model_validate(metadata["candidate"]), active_claim_ids=active)
            (destination / "SKILL.md").write_text(content, encoding="utf-8")
        except (KeyError, ValueError):
            shutil.copy2(skills_dir / skill_name / "SKILL.md", destination / "SKILL.md")
        active_claim_count += len(active)
        _write_json(destination / "applicability.json", {**metadata, "claims": [item for item in _dicts(metadata.get("claims")) if item.get("claim_id") in active]})
        available.append(skill_name)
    if not routed:
        policy = "all_active_skills"
    elif mount_selected_only:
        policy = f"deterministic_top_{top_k}_positive_overlap_mount_filter"
    else:
        policy = f"deterministic_top_{top_k}_recommendation_only"
    return {
        "available_skills": available,
        "eligible_skills": [name for name, _, _ in candidates],
        "selected_skills": recommended,
        "recommended_skills": recommended,
        "routing_policy": policy,
        "routing_records": routing_records,
        "suppressed_skills": suppressed_skills,
        "active_claim_count": active_claim_count,
        "suppressed_claim_count": suppressed_claim_count,
        "suppressed_claim_ids": sorted(suppressed),
    }


def curate_and_apply_positive_skills(
    *,
    curator: LLMPositiveSkillCurator,
    skills_dir: Path,
    family_name: str,
    task_name: str,
    task_position: int,
    iteration: int,
    analyzer_report: dict[str, Any],
    log_prefix: Path,
) -> SkillCuratorRun:
    manager = SkillBankManager(skills_dir)
    lifecycle_decisions = manager.apply_claim_feedback(analyzer_report, task_name=task_name, iteration=iteration)
    response, report, usage = curator.curate(
        skills_dir=skills_dir,
        family_name=family_name,
        task_name=task_name,
        task_position=task_position,
        analyzer_report=analyzer_report,
        log_prefix=log_prefix,
    )
    content_decisions = manager.apply_candidates(response, task_name=task_name, iteration=iteration, analyzer_report=analyzer_report)
    decisions = lifecycle_decisions + content_decisions
    report["pipeline_schema_version"] = 2
    report["skill_pipeline_schema_version"] = 3
    report["decisions"] = [asdict(decision) for decision in decisions]
    report["token_usage"] = token_usage_dict(usage)
    report_path = log_prefix.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return SkillCuratorRun(report=report, decisions=decisions, token_usage=usage)
