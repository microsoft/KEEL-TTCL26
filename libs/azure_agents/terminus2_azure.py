# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import asyncio
import shlex
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, indent, tostring
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM
from harbor.models.trial.paths import EnvironmentPaths
from libs.azure_agents.azure_llm import AzureOpenAILLM
from libs.azure_agents.terminus_json_parser import TolerantTerminusJSONParser


class ResilientTmuxSession(TmuxSession):
    _TOOL_INSTALL_TIMEOUT_SEC = 300
    _TOOL_INSTALL_BUDGET_SEC = 330
    _MAX_SESSION_RESTARTS = 2
    _MAX_INITIAL_SESSION_STARTS = 3

    async def start(self) -> None:
        for attempt in range(1, self._MAX_INITIAL_SESSION_STARTS + 1):
            try:
                await super().start()
                return
            except RuntimeError as exc:
                if "Failed to start tmux session" not in str(exc):
                    raise
                if await super().is_session_alive():
                    return
                if attempt == self._MAX_INITIAL_SESSION_STARTS:
                    raise
                await asyncio.sleep(attempt)

    async def send_keys(self, keys: str | list[str], block: bool = False, min_timeout_sec: float = 0.0, max_timeout_sec: float = 180.0) -> None:
        try:
            await super().send_keys(keys=keys, block=block, min_timeout_sec=min_timeout_sec, max_timeout_sec=max_timeout_sec)
        except RuntimeError as exc:
            if not any(marker in str(exc).casefold() for marker in ("no server running", "can't find session", "session not found")):
                raise
            self._logger.warning("Terminus-2 shell exited; restarting tmux session and retrying the pending action.")
            self._previous_buffer = None
            await self.start()
            await super().send_keys(keys=keys, block=block, min_timeout_sec=min_timeout_sec, max_timeout_sec=max_timeout_sec)

    async def is_session_alive(self) -> bool:
        if await super().is_session_alive():
            return True
        restarts = getattr(self, "_restart_count", 0)
        if restarts >= self._MAX_SESSION_RESTARTS:
            return False
        self._restart_count = restarts + 1
        self._previous_buffer = None
        try:
            await self.start()
        except Exception as exc:
            self._logger.warning("Unable to restart Terminus-2 tmux session: %s", exc)
            return False
        return await super().is_session_alive()

    async def _attempt_tmux_installation(self) -> None:
        try:
            await asyncio.wait_for(super()._attempt_tmux_installation(), timeout=self._TOOL_INSTALL_BUDGET_SEC)
        except (TimeoutError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "timed out" not in str(exc).lower():
                raise
            verify = await self.environment.exec(command="tmux -V", user="root", timeout_sec=60)
            if verify.return_code != 0:
                raise RuntimeError("Terminus-2 could not install tmux within the extended parallel-start budget") from exc


class AzureTerminus2(Terminus2):
    def __init__(self, *args: Any, prompt_template_path: str, **kwargs: Any) -> None:
        self._custom_prompt_template_path = Path(prompt_template_path).expanduser().resolve()
        super().__init__(*args, **kwargs)

    def _get_prompt_template_path(self) -> Path:
        return self._custom_prompt_template_path

    def _get_parser(self):
        return TolerantTerminusJSONParser()

    async def _build_skills_section(self, environment: BaseEnvironment) -> str | None:
        if not self.skills_dir or not await environment.is_dir(self.skills_dir):
            return None
        result = await environment.exec(f"find {shlex.quote(self.skills_dir)} -mindepth 2 -maxdepth 2 -name SKILL.md -type f | sort", timeout_sec=60)
        if result.return_code != 0 or not result.stdout or not result.stdout.strip():
            return None
        root = Element("available_skills")
        for path in result.stdout.strip().splitlines():
            cat_result = await environment.exec(f"cat {shlex.quote(path)}", timeout_sec=60)
            frontmatter = self._parse_skill_frontmatter(cat_result.stdout) if cat_result.return_code == 0 and cat_result.stdout else None
            if frontmatter is not None:
                skill = SubElement(root, "skill")
                SubElement(skill, "name").text = frontmatter["name"]
                SubElement(skill, "description").text = frontmatter["description"]
                SubElement(skill, "location").text = path
        if len(root) == 0:
            return None
        indent(root, space="  ")
        return "\n" + tostring(root, encoding="unicode")

    async def setup(self, environment: BaseEnvironment) -> None:
        self._session = ResilientTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    def _init_llm(self, model_name: str, reasoning_effort: str | None, **_: Any) -> BaseLLM:
        return AzureOpenAILLM(model_name=model_name, reasoning_effort=reasoning_effort)
