# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from libs.azure_agents.azure_llm import AzureOpenAILLM
from method_runner.checks import CheckPartition, VerifierEvidence
from method_runner.llm_json import call_structured_llm
from method_runner.token_usage import TokenUsage

ANALYZER_SYSTEM_PROMPT = """You are the diagnostic Analyzer in a verifier-guided
lifelong-learning experiment. You do not operate a terminal and you do not edit
skills or memory. Analyze only the supplied evidence.

Rules:
1. A passed check proves only the artifact property tested by that check. Do not
   label every earlier action as useful.
2. Separate observable failure facts from causal hypotheses.
3. Every positive behavior and failure hypothesis must cite supplied trajectory
   or artifact/verifier evidence references.
4. Use confidence "high" only when the causal link is explicit, "medium" when
   multiple evidence sources support it, "low" for a plausible but unconfirmed
   hypothesis, and "insufficient_evidence" when no grounded diagnosis is possible.
5. Do not invent missing actions, artifact contents, expected values, or causes.
6. Failure diagnoses and repair hypotheses may contain task-specific details
   when those details are directly supported by the evidence. A later Family
   Repair Memory Curator, not you, decides what can be generalized and persisted.
7. Return exactly one JSON object matching the requested schema, without Markdown.
8. For every passed check, cite one to three specific trajectory action steps
   that created or verified the passing property and summarize those actions in
   grounded_actions. Do not cite the user instruction, final completion claim,
   or trajectory.document as procedural evidence. Never leave a passed check's
   trajectory_step_refs or grounded_actions empty.
9. Express every positive item as one or more narrow behavior_claims. A claim
   must stay within the scope of its passed check: existence checks support only
   creation/path claims; structure checks support only tested structure;
   formatting checks support only tested formatting; semantic checks may support
   only the directly connected computation.
10. For every active claim from a skill the Actor actually read, emit exactly
   one skill_claim_feedback item: supported, contradicted, or not_evaluable.
   A skill read plus a task failure is not a contradiction. Contradiction
   requires an observed action showing the claim was followed and a failed check
   whose evidence directly opposes it.
11. A passed check establishes an observable outcome, not necessarily the
    algorithm that produced it. Mark procedural_attribution "direct" only when
    the verifier specifically establishes the claimed procedure. Otherwise use
    "compatible" or "insufficient" and explain why in attribution_reason.
12. Mark causal_attribution "direct" only when verifier or artifact evidence
    establishes the failure mechanism. Use "hypothesis" for a grounded but
    unconfirmed cause and "insufficient" when no causal diagnosis is supported.
    Do not attach a prescriptive repair hypothesis to insufficient evidence.
13. An exact-membership or exact-value check is semantic evidence for the tested
    result (for example, the exact set difference), separate from existence and
    ordering properties. Emit a narrow semantic_computation behavior claim for
    that result, but keep procedural_attribution "compatible" unless the verifier
    establishes the algorithm itself.
"""
ClaimScope = Literal["existence", "structure", "formatting", "transformation", "semantic_computation", "preservation", "verification"]
Confidence = Literal["high", "medium", "low", "insufficient_evidence"]
CLAIM_ID_CHARS = r"[^A-Za-z0-9_.:\-\[\]]+"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BehaviorClaim(StrictModel):
    claim_id: str = Field(pattern=r"^[A-Za-z0-9_.:\-\[\]]+$")
    claim: str = Field(min_length=1)
    claim_scope: ClaimScope
    supported_properties: list[str] = Field(min_length=1)
    procedural_attribution: Literal["direct", "compatible", "insufficient"] = "compatible"
    attribution_reason: str = Field(default="The check verifies an observable outcome but does not directly establish the procedure.", min_length=1)

    @field_validator("claim_id", mode="before")
    @classmethod
    def normalize_generated_claim_id(cls, value: Any) -> Any:
        return re.sub(CLAIM_ID_CHARS, "-", value).strip("-") if isinstance(value, str) else value


class PositiveBehavior(StrictModel):
    check: str
    status: Literal["verifier_supported_positive"] = "verifier_supported_positive"
    support: Literal["1/1"] = "1/1"
    supported_behavior: str = Field(min_length=1)
    why_supported: str = Field(min_length=1)
    trajectory_step_refs: list[str] = Field(min_length=1)
    grounded_actions: list[str] = Field(min_length=1)
    behavior_claims: list[BehaviorClaim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: Confidence
    family_generalizable: bool


class RepairHypothesis(StrictModel):
    recommended_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class FailureEvidence(StrictModel):
    check: str
    status: Literal["failed_check"] = "failed_check"
    observed_failure: str = Field(min_length=1)
    likely_cause: str | None
    trajectory_step_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: Confidence
    repair_hypothesis: RepairHypothesis | None
    validation_condition: str = Field(min_length=1)
    causal_attribution: Literal["direct", "hypothesis", "insufficient"] = "hypothesis"


class UnknownEvidence(StrictModel):
    check: str
    reason: str = Field(min_length=1)


class SkillClaimFeedback(StrictModel):
    skill_name: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    outcome: Literal["supported", "contradicted", "not_evaluable"]
    checks: list[str] = Field(default_factory=list)
    trajectory_step_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    confidence: Confidence


class SemanticAnalysis(StrictModel):
    analysis_summary: str = Field(min_length=1)
    positive_behavior_evidence: list[PositiveBehavior]
    failure_evidence: list[FailureEvidence]
    unknown_evidence: list[UnknownEvidence]
    skill_claim_feedback: list[SkillClaimFeedback] = Field(default_factory=list)


@dataclass
class AnalyzerRun:
    report: dict[str, Any]
    token_usage: TokenUsage


def _read_text(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= max_chars else text[:max_chars] + f"\n...[truncated after {max_chars} characters]"


def _load_json_or_text(path: Path, max_chars: int) -> Any:
    text = _read_text(path, max_chars)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return text


def _truncate_middle(text: str, max_chars: int, marker: str = "\n...[truncated]...\n") -> str:
    if len(text) <= max_chars:
        return text
    available = max(0, max_chars - len(marker))
    head = available * 3 // 4
    return text[:head] + marker + text[len(text) - (available - head) :] if available else text[:max_chars]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _compact_trajectory(trajectory: Any, max_chars: int) -> Any:
    if not isinstance(trajectory, dict) or len(_dumps(trajectory)) <= max_chars:
        return trajectory
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return _truncate_middle(_dumps(trajectory), max_chars)
    compact: dict[str, Any] = {key: trajectory[key] for key in ("schema_version", "session_id", "agent") if key in trajectory}
    compact["steps"] = [{key: step[key] for key in ("step_id", "source", "role") if key in step} if isinstance(step, dict) else {} for step in steps]
    contents = []
    for step in steps:
        shown = {key: step[key] for key in ("message", "tool_calls", "observation") if step.get(key) not in (None, "", [], {})} if isinstance(step, dict) else step
        if not shown and isinstance(step, dict):
            shown = {key: value for key, value in step.items() if key not in {"metrics", "extra", "timestamp", "model_name"}}
        contents.append(_dumps(shown))
    structural = len(_dumps(compact))
    budget = max(0, max_chars - structural - 256)
    per_step = budget // max(1, len(steps))
    for _ in range(5):
        for compact_step, content in zip(compact["steps"], contents, strict=True):
            compact_step["content"] = _truncate_middle(content, per_step, "\n...[trajectory step truncated for analyzer]...\n")
        size = len(_dumps(compact))
        if size <= max_chars or per_step == 0:
            break
        per_step = int(per_step * budget / max(1, size - structural) * 0.95)
    return compact


def _trajectory_payload(trial: Path, max_chars: int) -> dict[str, Any]:
    path = trial / "agent" / "trajectory.json"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    try:
        content = _compact_trajectory(json.loads(text), max_chars) if text else None
    except json.JSONDecodeError:
        content = _truncate_middle(text, max_chars)
    return {"path": str(path), "format": "json", "content": content}


def _trajectory_steps(trajectory: dict[str, Any]) -> list[Any]:
    content = trajectory.get("content")
    return content["steps"] if isinstance(content, dict) and isinstance(content.get("steps"), list) else []


def _trajectory_action_refs(trajectory: dict[str, Any]) -> list[str]:
    return [
        f"trajectory.step.{index}"
        for index, step in enumerate(_trajectory_steps(trajectory))
        if isinstance(step, dict) and str(step.get("source") or step.get("role") or "").casefold() != "user"
    ]


def _available_evidence_refs(payload: dict[str, Any]) -> list[str]:
    refs = [f"verifier.check.{item['name']}" for item in payload["check_evidence"]]
    if payload["task"]["instruction"]:
        refs.append("task.instruction")
    if payload["verifier"]["stdout"]:
        refs.append("verifier.stdout")
    if payload["artifact"]["manifest"] is not None:
        refs.append("artifact.manifest")
    if payload["trajectory"]["content"] not in (None, ""):
        refs.append("trajectory.document")
    refs.extend(f"trajectory.step.{index}" for index in range(len(_trajectory_steps(payload["trajectory"]))))
    return list(dict.fromkeys(refs))


REQUIREMENT_LINE_RE = re.compile(
    r"(?i)(must|required|exact|only|do not|don't|save|write|create|keep|preserve|without changing|schema|formula|filename|worksheet|sheet|column|row|cell)"
)
REQUIREMENT_TYPES = (
    ("schema", r"(?i)\b(schema|key|json|column|header)\b"),
    ("structure", r"(?i)\b(sheet|worksheet|row|cell|slide|order|range)\b"),
    ("preservation", r"(?i)\b(keep|preserve|do not change|without changing)\b"),
)


def build_requirement_ledger(instruction: str, *, max_items: int = 80) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in instruction.splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", raw_line).strip()
        if not line or line.startswith("```") or not REQUIREMENT_LINE_RE.search(line):
            continue
        normalized = re.sub(r"\s+", " ", line)
        if normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        requirement_type = (
            "path" if "/" in normalized or "\\" in normalized else next((t for t, pattern in REQUIREMENT_TYPES if re.search(pattern, normalized)), "exact_requirement")
        )
        items.append({"requirement_id": f"requirement.{len(items)}", "type": requirement_type, "exact_requirement": normalized[:500]})
        if len(items) >= max_items:
            break
    return items


def _load_skill_context(skills_dir: Path | None, skill_telemetry: dict[str, Any] | None) -> dict[str, Any]:
    telemetry = skill_telemetry or {}
    available = sorted(str(name) for name in telemetry.get("available_skills", []) if str(name))
    read = sorted(str(name) for name in telemetry.get("used_available_skills", []) if str(name))
    claims: list[dict[str, Any]] = []
    for skill_name in read if skills_dir is not None else []:
        try:
            metadata = json.loads((skills_dir / skill_name / "applicability.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in metadata.get("claims") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "provisional")
            if status in {"provisional", "confirmed"} and isinstance(item.get("claim_id"), str) and isinstance(item.get("claim"), str):
                claims.append({"skill_name": skill_name, "claim_id": item["claim_id"], "claim": item["claim"], "claim_scope": item.get("claim_scope"), "status": status})
    return {"available_skills": available, "read_skills": read, "active_claims": claims}


def _check_brief(evidence: VerifierEvidence, name: str) -> dict[str, Any]:
    check = evidence.checks.get(name)
    if check is None:
        return {"name": name, "status": "unknown"}
    return {
        "evidence_ref": f"verifier.check.{name}",
        "name": check.name,
        "status": check.status,
        "first_assertion": check.first_assertion,
        "expected": check.expected,
        "actual": check.actual,
        "message": check.message,
    }


def build_analyzer_input(
    *,
    family_name: str,
    task_path: Path,
    task_position: int,
    attempt: VerifierEvidence,
    partition: CheckPartition,
    trial: Path,
    repair_memory_path: Path | None,
    skills_dir: Path | None,
    skill_telemetry: dict[str, Any] | None,
    max_trajectory_chars: int,
) -> dict[str, Any]:
    check_names = partition.passed + partition.failed + partition.unresolved + partition.infrastructure_error
    stdout_path = trial / "verifier" / "test-stdout.txt"
    manifest_path = trial / "artifacts" / "manifest.json"
    instruction = _read_text(task_path / "instruction.md", 20_000)
    payload = {
        "task": {
            "family": family_name,
            "task_name": task_path.name,
            "task_position": task_position,
            "instruction": instruction,
            "requirement_ledger": build_requirement_ledger(instruction),
        },
        "attempt": {
            "trial_dir": str(trial),
            "reward": attempt.reward,
            "passed": attempt.passed,
            "total": attempt.total,
            "infrastructure_error": attempt.infrastructure_error,
        },
        "partition": asdict(partition),
        "check_evidence": [_check_brief(attempt, name) for name in check_names],
        "verifier": {"stdout_path": str(stdout_path), "stdout": _read_text(stdout_path, 20_000), "ctrf_path": str(trial / "verifier" / "ctrf.json")},
        "trajectory": _trajectory_payload(trial, max_trajectory_chars),
        "artifact": {"manifest_path": str(manifest_path), "manifest": _load_json_or_text(manifest_path, 20_000)},
        "current_repair_memory": _read_text(repair_memory_path, 12_000) if repair_memory_path is not None else "",
        "skill_context": _load_skill_context(skills_dir, skill_telemetry),
    }
    payload["available_trajectory_action_refs"] = _trajectory_action_refs(payload["trajectory"])
    payload["available_evidence_refs"] = _available_evidence_refs(payload)
    return payload


def _infer_claim_scope(check_name: str, behavior: str) -> ClaimScope:
    text = f"{check_name} {behavior}".casefold()
    for scope, pattern in (
        ("existence", r"\b(exists?|file_exists|output_exists)\b"),
        ("formatting", r"\b(format|font|style|fill|color|decimal)\b"),
        ("preservation", r"\b(preserve|unchanged|survive)\b"),
        ("structure", r"\b(schema|structure|sheet|column|header|row_count|order)\b"),
        ("semantic_computation", r"\b(value|answer|formula|range|metric|calculation|ranking)\b"),
    ):
        if re.search(pattern, text):
            return scope
    return "verification"


def _ensure_behavior_claims(analysis: SemanticAnalysis) -> SemanticAnalysis:
    for item in analysis.positive_behavior_evidence:
        if not item.behavior_claims:
            safe_check = re.sub(r"[^A-Za-z0-9_.-]+", "-", item.check).strip("-")
            item.behavior_claims = [
                BehaviorClaim(
                    claim_id=f"positive.{safe_check or 'check'}.claim.1",
                    claim=item.supported_behavior,
                    claim_scope=_infer_claim_scope(item.check, item.supported_behavior),
                    supported_properties=[item.supported_behavior],
                )
            ]
    return analysis


def _validate_analysis(analysis: SemanticAnalysis, partition: CheckPartition, analyzer_input: dict[str, Any]) -> None:
    allowed_refs = set(analyzer_input["available_evidence_refs"])
    allowed_action_refs = set(analyzer_input["available_trajectory_action_refs"])
    expected_unknown = partition.unresolved + partition.infrastructure_error
    problems: list[str] = []
    for label, actual, expected in (
        ("positive", [i.check for i in analysis.positive_behavior_evidence], partition.passed),
        ("failure", [i.check for i in analysis.failure_evidence], partition.failed),
        ("unknown", [i.check for i in analysis.unknown_evidence], expected_unknown),
    ):
        if sorted(actual) != sorted(expected):
            problems.append(f"{label} checks {actual!r} != {expected!r}")
    for label, items in (("positive", analysis.positive_behavior_evidence), ("failure", analysis.failure_evidence)):
        for item in items:
            invalid_refs = [ref for ref in [*item.evidence_refs, *item.trajectory_step_refs] if ref not in allowed_refs]
            if invalid_refs:
                problems.append(f"{label} check {item.check!r} has unknown evidence refs {invalid_refs!r}")
            if label != "positive":
                continue
            claim_ids = [claim.claim_id for claim in item.behavior_claims]
            if len(claim_ids) != len(set(claim_ids)):
                problems.append(f"positive check {item.check!r} has duplicate behavior claim IDs")
            invalid_action_refs = [ref for ref in item.trajectory_step_refs if ref not in allowed_action_refs]
            if invalid_action_refs:
                problems.append(
                    f"positive check {item.check!r} must cite Actor action steps; invalid refs {invalid_action_refs!r}; allowed action refs are {sorted(allowed_action_refs)!r}"
                )
            if any(not action.strip() for action in item.grounded_actions):
                problems.append(f"positive check {item.check!r} has blank grounded_actions")
    for item in analysis.failure_evidence:
        if item.causal_attribution == "direct" and (not item.likely_cause or item.confidence not in {"high", "medium"}):
            problems.append(f"direct causal attribution for failed check {item.check!r} requires a likely_cause and high or medium confidence")
        elif item.causal_attribution == "insufficient" and item.repair_hypothesis:
            problems.append(f"insufficient causal attribution for failed check {item.check!r} cannot include a prescriptive repair_hypothesis")
    active_claims = {
        (str(item.get("skill_name")), str(item.get("claim_id")))
        for item in analyzer_input["skill_context"]["active_claims"]
        if item.get("skill_name") and item.get("claim_id")
    }
    feedback_keys = [(item.skill_name, item.claim_id) for item in analysis.skill_claim_feedback]
    if sorted(feedback_keys) != sorted(active_claims):
        problems.append(f"skill claim feedback coverage {feedback_keys!r} != {sorted(active_claims)!r}")
    passed, failed = set(partition.passed), set(partition.failed)
    for item in analysis.skill_claim_feedback:
        tag = f"{item.skill_name}/{item.claim_id}"
        invalid_checks = [check for check in item.checks if check not in passed | failed | set(expected_unknown)]
        if invalid_checks:
            problems.append(f"skill feedback {tag} has unknown checks {invalid_checks!r}")
        invalid_refs = [ref for ref in [*item.evidence_refs, *item.trajectory_step_refs] if ref not in allowed_refs]
        if invalid_refs:
            problems.append(f"skill feedback {tag} has unknown refs {invalid_refs!r}")
        for outcome, allowed in (("supported", passed), ("contradicted", failed)):
            if item.outcome != outcome:
                continue
            if not item.checks or any(check not in allowed for check in item.checks):
                problems.append(f"{outcome} skill feedback {tag} must cite {'passed' if outcome == 'supported' else 'failed'} checks only")
            if outcome == "contradicted" and item.confidence not in {"high", "medium"}:
                problems.append(f"contradicted skill feedback {tag} must have high or medium confidence")
            if not item.trajectory_step_refs:
                problems.append(f"{outcome} skill feedback {tag} must cite an observed Actor action")
    if problems:
        raise ValueError("; ".join(problems))


def _referenced_positive_trajectory_steps(analysis: SemanticAnalysis, analyzer_input: dict[str, Any], max_chars: int = 4_000) -> dict[str, dict[str, Any]]:
    steps = _trajectory_steps(analyzer_input["trajectory"])
    resolved: dict[str, dict[str, Any]] = {}
    for ref in sorted({ref for item in analysis.positive_behavior_evidence for ref in item.trajectory_step_refs}):
        match = re.fullmatch(r"trajectory\.step\.(\d+)", ref)
        if not match or int(match.group(1)) >= len(steps) or not isinstance(steps[int(match.group(1))], dict):
            continue
        step = steps[int(match.group(1))]
        content = str(step.get("content") or step.get("message") or "")
        resolved[ref] = {
            "role": step.get("role") or step.get("source"),
            "content_excerpt": _truncate_middle(content, max_chars, "\n...[step excerpt truncated]...\n"),
            "truncated": len(content) > max_chars,
        }
    return resolved


class LLMAnalyzer:
    def __init__(
        self,
        *,
        model_name: str = "claude-opus-4.6",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        api_base: str | None = None,
        transport: str = "api",
        max_trajectory_chars: int = 60_000,
        max_output_tokens: int = 16_384,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_trajectory_chars = max_trajectory_chars
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

    def analyze(
        self,
        *,
        family_name: str,
        task_path: Path,
        task_position: int,
        attempt: VerifierEvidence,
        partition: CheckPartition,
        trial: Path,
        repair_memory_path: Path | None,
        log_prefix: Path,
        skills_dir: Path | None = None,
        skill_telemetry: dict[str, Any] | None = None,
    ) -> AnalyzerRun:
        analyzer_input = build_analyzer_input(
            family_name=family_name,
            task_path=task_path,
            task_position=task_position,
            attempt=attempt,
            partition=partition,
            trial=trial,
            repair_memory_path=repair_memory_path,
            skills_dir=skills_dir,
            skill_telemetry=skill_telemetry,
            max_trajectory_chars=self.max_trajectory_chars,
        )
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        log_prefix.with_suffix(".request.json").write_text(json.dumps(analyzer_input, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt = (
            "Analyze the evidence and return JSON matching this schema. Produce exactly "
            "one positive item for each passed check, exactly one failure item for each "
            "failed check, and exactly one unknown item for each unresolved or "
            "infrastructure-error check. Use check names exactly as supplied. Every "
            "evidence_refs and trajectory_step_refs entry must exactly match an ID in "
            "available_evidence_refs. Every passed check must cite at least one ID "
            "from available_trajectory_action_refs and describe the successful actions "
            "in grounded_actions. Every positive item must contain at least one narrow "
            "behavior_claim. For every behavior claim, explicitly distinguish a "
            "procedure directly established by the check from one merely compatible "
            "with the passing outcome. For every failure, explicitly distinguish a "
            "directly established cause from a hypothesis or insufficient evidence. "
            "Emit exactly one skill_claim_feedback item for each item "
            "in skill_context.active_claims.\n\n"
            + json.dumps(SemanticAnalysis.model_json_schema(), ensure_ascii=False)
            + "\n\nEVIDENCE INPUT:\n"
            + json.dumps(analyzer_input, ensure_ascii=False)
        )
        llm_run = call_structured_llm(
            client=self.client,
            model_type=SemanticAnalysis,
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            prompt=prompt,
            log_prefix=log_prefix,
            max_output_tokens=self.max_output_tokens,
            max_attempts=2,
            usage_source="azure_analyzer",
            model_name=self.model_name,
            value_transform=_ensure_behavior_claims,
            validator=lambda value: _validate_analysis(value, partition, analyzer_input),
        )
        report: dict[str, Any] = {
            "pipeline_schema_version": 2,
            "analyzer_model": self.model_name,
            "task_name": task_path.name,
            "family": family_name,
            "task_position": task_position,
            "attempt": analyzer_input["attempt"],
            "partition": asdict(partition),
        }
        analysis = llm_run.value
        if isinstance(analysis, SemanticAnalysis):
            report |= {
                "analysis_status": "complete",
                **analysis.model_dump(mode="json"),
                "referenced_trajectory_steps": _referenced_positive_trajectory_steps(analysis, analyzer_input),
            }
        else:
            unknown_checks = partition.passed + partition.failed + partition.unresolved + partition.infrastructure_error
            report |= {
                "analysis_status": "failed",
                "analysis_summary": "Analyzer produced no schema-valid grounded result.",
                "positive_behavior_evidence": [],
                "failure_evidence": [],
                "unknown_evidence": [{"check": name, "reason": "LLM Analyzer unavailable or invalid."} for name in unknown_checks],
                "skill_claim_feedback": [],
            }
        report["analyzer_errors"] = llm_run.errors
        return AnalyzerRun(report=report, token_usage=llm_run.token_usage)


def write_analyzer_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
