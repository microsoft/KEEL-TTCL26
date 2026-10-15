# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import json
from pathlib import Path


def ensure_standard_trajectory(trial_dir: Path) -> Path | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if trajectory_path.exists():
        return trajectory_path
    raw_trajectory = trial_dir / "agent" / "trajectory.md"
    if not raw_trajectory.exists():
        return None
    step = {
        "step_id": 1,
        "source": "agent",
        "message": raw_trajectory.read_text(encoding="utf-8", errors="replace"),
        "extra": {"raw_trajectory": "trajectory.md"},
    }
    trajectory = {
        "schema_version": "ATIF-v1.0",
        "session_id": trial_dir.name,
        "agent": {"name": "keel-runner", "version": "local-build"},
        "steps": [step],
    }
    trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    return trajectory_path
