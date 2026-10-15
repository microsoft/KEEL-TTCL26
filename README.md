# KEEL

Code for **KEEL**, a failure-aware, single-trajectory skill-evolution method for terminal-based coding agents, evaluated on the [SkillFlow](https://arxiv.org/abs/2604.17308) executable benchmark. The agent runs each task once; failures are analyzed and distilled into reusable skills and repair memory that carry forward to later tasks in the same family.

This repository is a research code release containing only the main experiment: a Terminus-2 actor with GPT-5.4 or GPT-5.6-Sol served through Azure OpenAI, over 166 tasks in 20 families. Tasks in a family run in order, sharing learned skills and repair memory; families run in parallel. It is provided as-is to support reproducibility and is not a supported product.

**Third-party code.** No third-party source is bundled. The code depends on [Harbor](https://github.com/laude-institute/harbor) (Apache-2.0) and its Terminus-2 agent, which `libs/azure_agents` and `libs/terminus_env` subclass, plus the Python packages pinned in `uv.lock`. See [NOTICE.md](NOTICE.md).

**Benchmark tasks are not included.** Download the SkillFlow tasks from [Hugging Face](https://huggingface.co/datasets/zhang-ziao/SkillFlow-Task) (follow its license) and place them under `test_tasks/test_tasks/<Family>/<task>/` (the paths listed in `configs/*.yaml`). Each task needs `instruction.md`, a task config, `environment/`, and `tests/`.

## 1. Install

Use Bash (Git Bash on Windows). Install Git, Docker (running, Linux containers) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/microsoft/KEEL-TTCL26.git
cd KEEL-TTCL26
uv python install 3.12
uv sync --locked --python 3.12
docker version
```

Task images are built from the task Dockerfiles when the experiment runs.

## 2. Configure Azure OpenAI

Create `.env` in the repository root (shell variables take precedence):

```dotenv
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Use the endpoint of your Azure OpenAI resource. Your resource needs deployments named `gpt-5.4` and/or `gpt-5.6-sol`; otherwise edit `agents[0].model_name` and `azure_api_version` in the YAML config and pass matching `--analyzer-model`/`--curator-model`. Never commit `.env` (it is git-ignored).

| Model | Config | Reasoning effort |
|---|---|---|
| GPT-5.4 | `configs/method_terminus2_gpt54_all.yaml` | `medium` |
| GPT-5.6-Sol | `configs/method_terminus2_gpt56sol_all.yaml` | `high` |

The actor's effort comes from its YAML; `--reasoning-effort` sets the analyzer and curators and should match.

## 3. Check the setup

Dry run (no model calls or containers), then a one-task smoke test:

```bash
uv run --locked python -m method_runner.failure_aware_runner \
  --config configs/method_terminus2_gpt54_all.yaml \
  --run-root-dir "runs/plan-gpt54-$(date +%Y%m%d-%H%M%S)" \
  --num-iterations 1 --analyzer-model gpt-5.4 --curator-model gpt-5.4 \
  --reasoning-effort medium --dry-run

uv run --locked python -m method_runner.failure_aware_runner \
  --config configs/method_terminus2_gpt54_all.yaml \
  --run-root-dir "runs/smoke-gpt54-$(date +%Y%m%d-%H%M%S)" \
  --family Cross-Format-Data-Reconciliation --max-tasks 1 \
  --num-iterations 1 --analyzer-model gpt-5.4 --curator-model gpt-5.4 \
  --reasoning-effort medium
```

## 4. Run the full experiment

Use a new output directory per experiment; reusing one reuses completed work.

```bash
RUN_DIR="runs/keel-azure-gpt54-$(date +%Y%m%d-%H%M%S)"
uv run --locked python -m method_runner.run_family_workers \
  --config configs/method_terminus2_gpt54_all.yaml \
  --run-root-dir "${RUN_DIR:?Set RUN_DIR before running}" \
  --workers 8 --num-iterations 1 \
  --analyzer-model gpt-5.4 --curator-model gpt-5.4 --reasoning-effort medium
```

For GPT-5.6-Sol use `configs/method_terminus2_gpt56sol_all.yaml`, `--analyzer-model gpt-5.6-sol --curator-model gpt-5.6-sol` and `--reasoning-effort high`.

`--workers 8` runs up to eight family workers; lower it if Docker or Azure quota is limited. `--num-iterations 1` is the main protocol: one actor rollout per task (infrastructure errors are retried once; verifier failures are not).

## 5. Results

```text
runs/<run-directory>/
  family_worker_status.json   # "status": "complete" when finished
  method_summary.json         # aggregate metrics; completed_tasks = tasks passed
  completion_rates.csv, task_test_results.csv, token_usage.csv, skill_usage.csv
  shards/worker_NN/           # worker.log, harbor_jobs/, skill_banks/, repair_memory/
```

## Contributing

This project welcomes contributions and suggestions. See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are subject to the Microsoft Contributor License Agreement. This project has adopted the [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md). For security issues see [SECURITY.md](SECURITY.md); for help see [SUPPORT.md](SUPPORT.md).

## License

Released under the [MIT License](LICENSE.txt).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
