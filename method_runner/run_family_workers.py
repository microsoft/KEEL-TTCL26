# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import yaml
from method_runner.failure_aware_runner import build_summary, write_reports

ROOT_DIR = Path(__file__).resolve().parents[1]


def family_names_from_config(config_path: Path) -> list[str]:
    datasets = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("datasets") or []
    return list(dict.fromkeys(Path(str(dataset["path"])).name for dataset in datasets if dataset.get("path")))


def partition_families(families: list[str], worker_count: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(min(worker_count, len(families)))]
    for index, family in enumerate(families):
        shards[index % len(shards)].append(family)
    return shards


def merge_shard_summaries(run_root: Path, shard_dirs: list[Path]) -> dict[str, Any]:
    summaries = [json.loads((shard_dir / "method_summary.json").read_text(encoding="utf-8")) for shard_dir in shard_dirs]
    tasks = sorted((task for summary in summaries for task in summary["tasks"]), key=lambda task: (task["family"], task["task_name"]))
    first = summaries[0]
    merged = build_summary(
        tasks,
        job_name="keel-single-trajectory-family-workers",
        model=first["model"],
        analyzer_model=first["analyzer_model"],
        curator_model=first["curator_model"],
        num_iterations=1,
        family_workers=len(shard_dirs),
        planned_families=[family for summary in summaries for family in summary["planned_families"]],
        shards=[str(path) for path in shard_dirs],
    )
    write_reports(run_root, merged)
    return merged


def write_status(run_root: Path, status: dict[str, Any]) -> int:
    (run_root / "family_worker_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent SkillFlow families in parallel")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--run-root-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-iterations", type=int, default=1, choices=[1], help="Single Actor rollout per task")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--analyzer-model", default=None)
    parser.add_argument("--curator-model", default=None)
    parser.add_argument("--family", action="append", default=[], help="Optional family subset; repeatable")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else (ROOT_DIR / args.config).resolve()
    run_root = args.run_root_dir if args.run_root_dir.is_absolute() else (ROOT_DIR / args.run_root_dir).resolve()
    families = family_names_from_config(config_path)
    if args.family:
        missing = sorted(set(args.family) - set(families))
        if missing:
            parser.error(f"Unknown families: {', '.join(missing)}")
        families = [family for family in families if family in set(args.family)]
    if not families or args.workers < 1:
        parser.error("No families selected or invalid --workers")
    shards = partition_families(families, args.workers)
    forwarded = [
        item
        for flag, value in (
            ("--reasoning-effort", args.reasoning_effort),
            ("--analyzer-model", args.analyzer_model),
            ("--curator-model", args.curator_model),
        )
        if value
        for item in (flag, value)
    ]
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    processes: list[tuple[subprocess.Popen[Any], Any, Path, list[str]]] = []
    run_root.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards, start=1):
        shard_dir = run_root / "shards" / f"worker_{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "method_runner.failure_aware_runner", "-c", str(config_path), "--run-root-dir", str(shard_dir), *forwarded]
        for family in shard:
            command += ["--family", family]
        log_handle = (shard_dir / "worker.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT_DIR, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((process, log_handle, shard_dir, shard))
        print(f"worker {index:02d} pid={process.pid} families={','.join(shard)}", flush=True)
    failures: list[dict[str, Any]] = []
    try:
        for process, log_handle, shard_dir, shard in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                failures.append({"shard": str(shard_dir), "families": shard, "return_code": return_code})
    except KeyboardInterrupt:
        for process, log_handle, _, _ in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise
    shard_dirs = [item[2] for item in processes]
    missing_summaries = [str(path / "method_summary.json") for path in shard_dirs if not (path / "method_summary.json").is_file()]
    if failures or missing_summaries:
        return write_status(
            run_root,
            {
                "summary_path": None,
                "status": "failed",
                "message": "Family-worker run is incomplete; no merged method summary was written",
                "workers": len(shards),
                "failures": failures,
                "missing_shard_summaries": missing_summaries,
            },
        )
    merged = merge_shard_summaries(run_root, shard_dirs)
    infrastructure_error_rate = merged["metrics"]["infrastructure_error_rate"]
    status = {
        "summary_path": str(run_root / "method_summary.json"),
        "status": "failed" if infrastructure_error_rate else "complete",
        "workers": len(shards),
        "total_tasks": merged["total_tasks"],
        "completed_tasks": merged["completed_tasks"],
        "completion_rate": merged["completion_rate"],
        "infrastructure_error_rate": infrastructure_error_rate,
    }
    if infrastructure_error_rate:
        status["message"] = "Family-worker run contains infrastructure errors"
    return write_status(run_root, status)


if __name__ == "__main__":
    raise SystemExit(main())
