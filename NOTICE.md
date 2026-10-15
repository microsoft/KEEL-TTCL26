# NOTICE

This repository does not redistribute third-party source code. It depends on the
following third-party projects, which are installed separately (see `pyproject.toml`
and `uv.lock`) and remain under their own licenses:

- **Harbor** (https://github.com/laude-institute/harbor), including its Terminus-2 agent,
  which `libs/azure_agents` subclasses and extends. Harbor is licensed under Apache-2.0.
- **Python packages** pinned in `uv.lock` (e.g. `openai`, `litellm`, `pydantic`, `numpy`,
  `matplotlib`, `openpyxl`, `pyyaml`, `tenacity`), each under its own license.

The **SkillFlow** benchmark tasks (Zhang et al., https://arxiv.org/abs/2604.17308;
https://huggingface.co/datasets/zhang-ziao/SkillFlow-Task) are **not** included in this
repository. Obtain them from their original source and follow their license terms.

Third-party package licenses can be reviewed with the tooling of your choice against `uv.lock`.
