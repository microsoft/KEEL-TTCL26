# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import yaml
from harbor.environments.docker.docker import DockerEnvironment

SKILLS_MOUNT_TARGET = "/root/.agents/skills"


class SharedSkillsDockerEnvironment(DockerEnvironment):
    def __init__(
        self,
        environment_dir: Path,
        *args,
        project_template_dir: str | None = None,
        shared_skills_root: str = "s",
        copy_task_skills: bool = False,
        **kwargs,
    ):
        self._project_template_dir = Path(project_template_dir).expanduser().resolve() if project_template_dir else None
        self._shared_skills_root = shared_skills_root
        self._copy_task_skills = copy_task_skills
        self._task_environment_dir = Path(environment_dir).resolve()
        super().__init__(self._task_environment_dir, *args, **kwargs)

    @property
    def _docker_compose_paths(self) -> list[Path]:
        override_path = self.trial_paths.trial_dir / "docker-compose-shared-skills.yaml"
        shared_skills_dir = self._prepare_shared_skills_dir()
        volumes = [f"{shared_skills_dir.resolve().as_posix()}:{SKILLS_MOUNT_TARGET}"]
        override_path.write_text(yaml.safe_dump({"services": {"main": {"volumes": volumes}}}, sort_keys=False))
        return [*super()._docker_compose_paths, override_path]

    def _prepare_shared_skills_dir(self) -> Path:
        group_name = self._task_group_name().strip().replace("/", "_").replace("\\", "_") or "adhoc"
        group_hash = hashlib.sha1(group_name.encode("utf-8")).hexdigest()[:8]
        shared_skills_dir = self.trial_paths.trial_dir.parent / self._shared_skills_root / group_hash
        shared_skills_dir.mkdir(parents=True, exist_ok=True)
        template_dir = self._project_template_dir
        if template_dir is not None and (template_dir / "skills").is_dir():
            template_dir = template_dir / "skills"
        self._copy_missing_tree(template_dir, shared_skills_dir)
        if self._copy_task_skills:
            self._copy_missing_tree(self._task_environment_dir / "skills", shared_skills_dir)
        return shared_skills_dir

    def _task_group_name(self) -> str:
        default_name = self.environment_name or "adhoc"
        config_path = self.trial_paths.config_path
        if not config_path.exists():
            return default_name
        try:
            config_data = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return default_name
        return (config_data.get("task") or {}).get("source") or default_name

    def _copy_missing_tree(self, source_dir: Path | None, target_dir: Path) -> None:
        if source_dir is None or not source_dir.is_dir():
            return
        for source_child in source_dir.iterdir():
            target_child = target_dir / source_child.name
            if source_child.is_dir():
                if not target_child.exists():
                    shutil.copytree(source_child, target_child)
                elif target_child.is_dir():
                    self._copy_missing_tree(source_child, target_child)
            elif not target_child.exists():
                shutil.copy2(source_child, target_child)
