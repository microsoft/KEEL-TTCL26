# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import yaml
from harbor import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.paths import TaskPaths
from harbor.trial.hooks import TrialHookEvent

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from libs.skill_evolution.patcher import ensure_standard_trajectory
from method_runner.actor import (
    build_augmented_instruction,
    build_iteration_job_config,
    copy_task_overlay,
    remove_tree,
    select_repair_memory,
)
from method_runner.analyzer import LLMAnalyzer, write_analyzer_report
from method_runner.checks import parse_verifier_evidence, partition_checks
from method_runner.repair_memory import FamilyRepairMemoryCurator, curate_family_repair_memory
from method_runner.skill_manager import (
    LLMPositiveSkillCurator,
    append_skill_decisions,
    curate_and_apply_positive_skills,
    materialize_active_skill_snapshot,
)
from method_runner.skill_telemetry import collect_skill_telemetry
from method_runner.token_usage import (
    COMPONENTS,
    TokenUsage,
    non_cached_input_tokens,
    sum_token_usage,
    token_usage_dict,
    token_usage_from_trial_dir,
    zero_usage,
)

SKILL_OPERATIONS = ("add_draft", "add_provisional", "merge", "confirm", "contradict", "retire", "reject")
MEMORY_OPERATIONS = ("add", "merge", "replace", "retire", "reject")
SKILL_USAGE_FIELDS = (
    "available_skill_count",
    "available_skills",
    "eligible_skill_count",
    "eligible_skills",
    "selected_skill_count",
    "selected_skills",
    "routing_policy",
    "skill_available",
    "skill_selected",
    "skill_used",
    "selected_skill_consumed",
    "used_skills",
    "skill_read_calls",
)


@dataclass
class TaskRecord:
    family: str
    task_name: str
    task_position: int
    completed: bool = False
    best_reward: float = 0.0
    best_tests_passed: int = 0
    total_tests: int = 0
    test_pass_rate: str = "0/0"
    infrastructure_error: str | None = None
    skill_operations: list[str] = field(default_factory=list)
    memory_operations: list[str] = field(default_factory=list)
    skill_telemetry: dict[str, Any] = field(default_factory=dict)
    repair_memory_prompt: dict[str, Any] = field(default_factory=dict)
    component_token_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_token_usage: dict[str, Any] = field(default_factory=dict)


def ensure_windows_utf8_process() -> None:
    if os.name != "nt" or sys.flags.utf8_mode:
        return
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    raise SystemExit(subprocess.call([sys.executable, "-m", "method_runner.failure_aware_runner", *sys.argv[1:]], env=environment))


def load_dotenv() -> None:
    for env_path in (ROOT_DIR / ".env", ROOT_DIR.parent / ".env", Path.cwd() / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.strip().partition("=")
            name, value = name.strip(), value.strip()
            if not separator or name.startswith("#") or not name:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(name, value)


def short_id(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def resolve_dataset_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else ROOT_DIR / expanded).resolve()


def is_valid_task_path(path: Path, *, disable_verification: bool) -> bool:
    task_paths = TaskPaths(path)
    if not (path.is_dir() and task_paths.instruction_path.is_file() and task_paths.config_path.is_file() and task_paths.environment_dir.is_dir()):
        return False
    return disable_verification or (task_paths.tests_dir.is_dir() and task_paths.discovered_test_path is not None)


def resolve_task_paths(dataset_root: Path, *, disable_verification: bool) -> list[Path]:
    paths = {path.name: path.resolve() for path in dataset_root.iterdir() if is_valid_task_path(path, disable_verification=disable_verification)}
    ranking_path = dataset_root / "ALL_TASK_DIFFICULTY_RANKING.json"
    ranking = [name for name in json.loads(ranking_path.read_text(encoding="utf-8")) if isinstance(name, str)] if ranking_path.is_file() else []
    return [paths[name] for name in ranking if name in paths] + [paths[name] for name in sorted(paths) if name not in set(ranking)]


def resolve_trial_dir(trial_uri: str) -> Path:
    if not trial_uri.startswith("file://"):
        return Path(trial_uri)
    text = unquote(urlparse(trial_uri).path)
    if os.name == "nt" and len(text) >= 3 and text[0] == "/" and text[2] == ":":
        text = text[1:]
    return Path(text)


async def run_trial_job(job_config: JobConfig) -> Path:
    trial_dirs: list[Path] = []

    async def on_trial_ended(event: TrialHookEvent) -> None:
        if event.result is not None:
            trial_dir = resolve_trial_dir(event.result.trial_uri)
            await asyncio.to_thread(ensure_standard_trajectory, trial_dir)
            trial_dirs.append(trial_dir)

    job = await Job.create(job_config)
    job.on_trial_ended(on_trial_ended)
    await job.run()
    if not trial_dirs:
        raise RuntimeError("Harbor job completed without emitting a trial result")
    return trial_dirs[-1]


def run_task(
    run_root: Path,
    base_config: JobConfig,
    models: dict[str, Any],
    family_name: str,
    dataset_root: Path,
    task_path: Path,
    task_position: int,
) -> TaskRecord:
    task_name = task_path.name
    task_key = f"{family_name}/{task_name}"
    skills_dir = run_root / "skill_banks" / family_name
    repair_memory_path = run_root / "repair_memory" / family_name / "MEMORY.md"
    overlay_base = run_root / "task_overlays" / short_id(task_key, 12)
    active_skills_dir = run_root / "skill_snapshots" / short_id(task_key, 12)
    job_name = f"fs1__{family_name[:10]}__{task_name[:12]}__{short_id(task_key)}".replace("/", "-").replace(" ", "_")
    job_dir = run_root / "harbor_jobs" / job_name

    skills_dir.mkdir(parents=True, exist_ok=True)
    base_instruction = (task_path / "instruction.md").read_text(encoding="utf-8")
    memory_text, memory_prompt = select_repair_memory(repair_memory_path, base_instruction)
    skill_availability = materialize_active_skill_snapshot(
        skills_dir,
        active_skills_dir,
        suppressed_claim_ids=set(memory_prompt["suppressed_skill_claim_ids"]),
        task_instruction=base_instruction,
        top_k=1,
    )
    copy_task_overlay(task_path, overlay_base / "rollout", build_augmented_instruction(base_instruction, memory_text, skill_availability))
    job_config = build_iteration_job_config(
        base_config,
        job_name=job_name,
        jobs_dir=run_root / "harbor_jobs",
        family_name=family_name,
        dataset_root=dataset_root,
        overlay_task=overlay_base / "rollout",
        shared_skills_dir=active_skills_dir,
    )
    if job_dir.exists():
        remove_tree(job_dir)
    trial_dir = asyncio.run(run_trial_job(job_config))

    skill_telemetry = collect_skill_telemetry(
        trial_dir, available_skills=[str(name) for name in skill_availability.get("available_skills", [])], availability_metadata=skill_availability
    )
    evidence = parse_verifier_evidence(trial_dir, task_path)
    analyzer_path = run_root / "analysis" / family_name / task_name / "analysis.json"
    analyzer_run = models["analyzer"].analyze(
        family_name=family_name,
        task_path=task_path,
        task_position=task_position,
        attempt=evidence,
        partition=partition_checks(evidence),
        trial=trial_dir,
        repair_memory_path=repair_memory_path,
        log_prefix=analyzer_path.with_suffix(""),
        skills_dir=active_skills_dir,
        skill_telemetry=skill_telemetry,
    )
    analyzer_run.report["token_usage"] = token_usage_dict(analyzer_run.token_usage)
    write_analyzer_report(analyzer_path, analyzer_run.report)
    memory_run = curate_family_repair_memory(
        curator=models["memory_curator"],
        memory_path=repair_memory_path,
        family_name=family_name,
        task_name=task_name,
        task_position=task_position,
        analyzer_report=analyzer_run.report,
        log_prefix=run_root / "memory_curations" / family_name / task_name / "curation",
    )
    skill_run = curate_and_apply_positive_skills(
        curator=models["skill_curator"],
        skills_dir=skills_dir,
        family_name=family_name,
        task_name=task_name,
        task_position=task_position,
        iteration=0,
        analyzer_report=analyzer_run.report,
        log_prefix=run_root / "skill_curations" / family_name / task_name / "curation",
    )
    append_skill_decisions(run_root / "skill_decisions.jsonl", skill_run.decisions)

    usages = {
        "actor": token_usage_from_trial_dir(trial_dir, model_name=models["actor_model"]),
        "analyzer": analyzer_run.token_usage,
        "repair_memory_curator": memory_run.token_usage,
        "positive_skill_curator": skill_run.token_usage,
    }
    return TaskRecord(
        family=family_name,
        task_name=task_name,
        task_position=task_position,
        completed=evidence.reward >= 1.0,
        best_reward=evidence.reward,
        best_tests_passed=evidence.passed,
        total_tests=evidence.total,
        test_pass_rate=f"{evidence.passed}/{evidence.total}",
        infrastructure_error=evidence.infrastructure_error,
        skill_operations=[decision.operation for decision in skill_run.decisions],
        memory_operations=[decision.operation for decision in memory_run.decisions],
        skill_telemetry=skill_telemetry,
        repair_memory_prompt=memory_prompt,
        component_token_usage={name: token_usage_dict(usage) for name, usage in usages.items()},
        total_token_usage=token_usage_dict(sum_token_usage(list(usages.values()), source="task_components")),
    )


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def sum_component_usages(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: token_usage_dict(
            sum_token_usage(
                [TokenUsage(**task["component_token_usage"].get(name, token_usage_dict(zero_usage()))) for task in tasks],
                source=f"{name}_aggregate",
            )
        )
        for name in COMPONENTS
    }


def build_run_metrics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)
    family_completion = {family: _rate(sum(t["completed"] for t in records), len(records)) for family, records in sorted(by_family.items())}
    family_inner = [_rate(sum(t["best_tests_passed"] for t in records), sum(t["total_tests"] for t in records)) for records in by_family.values()]
    telemetry = [task["skill_telemetry"] for task in tasks if task["skill_telemetry"]]
    available = sum(1 for item in telemetry if item.get("skill_available"))
    used_when_available = sum(1 for item in telemetry if item.get("skill_available") and item.get("skill_used"))
    skill_ops = Counter(operation for task in tasks for operation in task["skill_operations"])
    memory_ops = Counter(operation for task in tasks for operation in task["memory_operations"])
    return {
        "task_completion_micro": _rate(sum(t["completed"] for t in tasks), len(tasks)),
        "task_completion_macro_families": _rate(sum(family_completion.values()), len(family_completion)),
        "family_completion_rates": family_completion,
        "inner_test_pass_micro": _rate(sum(t["best_tests_passed"] for t in tasks), sum(t["total_tests"] for t in tasks)),
        "inner_test_pass_macro_families": _rate(sum(family_inner), len(family_inner)),
        "skill_operations": {operation: skill_ops[operation] for operation in SKILL_OPERATIONS},
        "memory_operations": {operation: memory_ops[operation] for operation in MEMORY_OPERATIONS},
        "skill_usage": {
            "definition": "A skill is used when the Actor reads its SKILL.md file or invokes a native Skill tool; listings do not count.",
            "task_iterations": len(telemetry),
            "available_task_iterations": available,
            "used_task_iterations": sum(1 for item in telemetry if item.get("skill_used")),
            "used_when_available_task_iterations": used_when_available,
            "availability_conditioned_usage_rate": _rate(used_when_available, available) if available else None,
            "skill_read_calls": sum(int(item.get("skill_read_calls") or 0) for item in telemetry),
        },
        "infrastructure_error_rate": _rate(sum(1 for task in tasks if task["infrastructure_error"]), len(tasks)),
    }


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_reports(run_root: Path, summary: dict[str, Any]) -> None:
    tasks = sorted(summary["tasks"], key=lambda task: (task["family"], task["task_name"]))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "method_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [["overall", "", summary["total_tasks"], summary["completed_tasks"], summary["completion_rate"]]]
    for family, rate in summary["metrics"]["family_completion_rates"].items():
        family_tasks = [task for task in tasks if task["family"] == family]
        rows.append(["family", family, len(family_tasks), sum(task["completed"] for task in family_tasks), rate])
    write_csv(run_root / "completion_rates.csv", ["scope", "family", "total_tasks", "completed_tasks", "completion_rate"], rows)
    write_csv(
        run_root / "task_test_results.csv",
        ["family", "task_name", "completed", "best_reward", "best_tests_passed", "total_tests", "test_pass_rate"],
        [[t["family"], t["task_name"], t["completed"], t["best_reward"], t["best_tests_passed"], t["total_tests"], t["test_pass_rate"]] for t in tasks],
    )

    def usage_row(scope: str, family: str, task_name: str, component: str, usage: dict[str, Any]) -> list[Any]:
        return [
            scope,
            family,
            task_name,
            component,
            usage.get("n_requests"),
            usage.get("n_input_tokens"),
            non_cached_input_tokens(usage.get("n_input_tokens"), usage.get("n_cache_tokens")),
            usage.get("n_cache_tokens"),
            usage.get("n_output_tokens"),
            usage.get("cost_usd"),
            usage.get("source"),
        ]

    usage_rows = [usage_row("overall_component", "", "", name, usage) for name, usage in summary["component_token_usage"].items()]
    usage_rows.append(usage_row("overall_total", "", "", "all", summary["total_token_usage"]))
    for task in tasks:
        usage_rows += [usage_row("task_component", task["family"], task["task_name"], name, usage) for name, usage in task["component_token_usage"].items()]
    write_csv(
        run_root / "token_usage.csv",
        [
            "scope",
            "family",
            "task_name",
            "component",
            "n_requests",
            "n_input_tokens",
            "n_non_cached_input_tokens",
            "n_cache_tokens",
            "n_output_tokens",
            "cost_usd",
            "source",
        ],
        usage_rows,
    )
    write_csv(
        run_root / "skill_usage.csv",
        ["family", "task_name", "task_position", *SKILL_USAGE_FIELDS],
        [
            [t["family"], t["task_name"], t["task_position"]]
            + [json.dumps(v, ensure_ascii=False) if isinstance(v := t["skill_telemetry"].get(name), list) else v for name in SKILL_USAGE_FIELDS]
            for t in tasks
        ],
    )


def build_summary(tasks: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    components = sum_component_usages(tasks)
    completed = sum(task["completed"] for task in tasks)
    return {
        "method": "KEEL",
        "protocol": "main_single_rollout_lifelong",
        **fields,
        "total_tasks": len(tasks),
        "completed_tasks": completed,
        "completion_rate": _rate(completed, len(tasks)),
        "metrics": build_run_metrics(tasks),
        "component_token_usage": components,
        "total_token_usage": token_usage_dict(sum_token_usage([TokenUsage(**usage) for usage in components.values()], source="run_components")),
        "tasks": tasks,
    }


def run_method(args: argparse.Namespace) -> dict[str, Any]:
    base_config = JobConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    actor_model = str(base_config.model_dump()["agents"][0].get("model_name") or "unknown")
    api_key_env, api_base = "AZURE_OPENAI_API_KEY", os.environ.get("AZURE_OPENAI_ENDPOINT")
    curator_model = args.curator_model or args.analyzer_model
    curator_kwargs = {"model_name": curator_model, "api_key_env": api_key_env, "api_base": api_base, "reasoning_effort": args.reasoning_effort}
    models = {
        "actor_model": actor_model,
        "analyzer": LLMAnalyzer(
            model_name=args.analyzer_model, api_key_env=api_key_env, api_base=api_base, max_output_tokens=8_192, reasoning_effort=args.reasoning_effort
        ),
        "memory_curator": FamilyRepairMemoryCurator(**curator_kwargs),
        "skill_curator": LLMPositiveSkillCurator(**curator_kwargs),
    }
    run_root = args.run_root_dir if args.run_root_dir.is_absolute() else (ROOT_DIR / args.run_root_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    datasets = [dataset for dataset in base_config.datasets if not args.family or Path(dataset.path).name in set(args.family)]
    tasks: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_root = resolve_dataset_path(Path(dataset.path))
        family_name = dataset_root.name
        task_paths = resolve_task_paths(dataset_root, disable_verification=bool(base_config.verifier.disable))[: args.max_tasks or None]
        planned.append({"family": family_name, "dataset_root": str(dataset_root), "n_tasks": len(task_paths), "tasks": [path.name for path in task_paths]})
        if args.dry_run:
            continue
        for task_position, task_path in enumerate(task_paths, start=1):
            record_path = run_root / "task_records" / family_name / f"{task_path.name}.json"
            if record_path.is_file():
                tasks.append(json.loads(record_path.read_text(encoding="utf-8")))
                continue
            record = asdict(run_task(run_root, base_config, models, family_name, dataset_root, task_path, task_position))
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tasks.append(record)
    summary = build_summary(
        tasks,
        job_name=base_config.job_name or "keel-method",
        model=actor_model,
        analyzer_model=args.analyzer_model,
        curator_model=curator_model,
        num_iterations=1,
        dry_run=args.dry_run,
        planned_families=planned,
    )
    if args.dry_run:
        summary["total_tasks"] = sum(item["n_tasks"] for item in planned)
    write_reports(run_root, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ensure_windows_utf8_process()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Failure-aware single-trajectory SkillFlow runner using a supported terminal Actor")
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/method_terminus2_gpt54_all.yaml"))
    parser.add_argument("--run-root-dir", type=Path, default=Path("jobs/keel-terminus2"))
    parser.add_argument("--num-iterations", type=int, default=1, choices=[1], help="Single Actor rollout per task")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--analyzer-model", default="claude-opus-4.6")
    parser.add_argument("--curator-model", default=None, help="Defaults to --analyzer-model")
    parser.add_argument("--family", action="append", default=[], help="Family directory name to run; repeatable")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve families/tasks and write summary without running Harbor")
    args = parser.parse_args(argv)
    load_dotenv()
    summary = run_method(args)
    run_root = args.run_root_dir if args.run_root_dir.is_absolute() else ROOT_DIR / args.run_root_dir
    print(
        json.dumps(
            {
                "summary_path": str(run_root / "method_summary.json"),
                "total_tasks": summary["total_tasks"],
                "completed_tasks": summary["completed_tasks"],
                "completion_rate": summary["completion_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
