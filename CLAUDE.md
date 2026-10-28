# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

StarVLA is a modular ("Lego-like") research codebase for Vision-Language-Action (VLA) models: pluggable backbones (VLMs, world models) × interchangeable action heads (FAST, OFT, flow-matching PI, GR00T dual-system), sharing one data interface, trainer, and deployment server.

## Setup & commands

```bash
conda create -n starVLA python=3.10 -y && conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional; falls back to SDPA if absent
pip install -e .
```

Tests are `unittest`-style under `tests/` (pytest is not a dependency). Run from the repo root so `starVLA` and `deployment` are importable; some tests need optional packages (`zmq`, `msgpack`):

```bash
python -m unittest discover -s tests -t .
python -m unittest tests.test_model_tools
python -m unittest tests.test_config_overrides.<TestClass>.<test_name>
```

Lint/format (Black + Ruff, line length 121). The repo has a historical lint backlog, so `make check` on the whole repo is **expected to fail** — only check files you touch, and do not commit drive-by reformatting:

```bash
FILES=$(git diff --name-only --diff-filter=ACMR origin/starVLA_dev | grep -E '\.py$')
black $FILES && python -m ruff check --fix $FILES
```

Smoke-test a single component standalone (each framework/dataloader file has a `__main__` that builds it from a YAML and runs a fake batch through `forward` / `predict_action`):

```bash
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml
python starVLA/dataloader/lerobot_datasets.py --config_yaml <yaml>
```

Training is launched with `accelerate` + DeepSpeed; any YAML key can be overridden as a `--dotted.key value` CLI arg (see `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` for a full example):

```bash
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  starVLA/training/train_starvla.py --config_yaml <yaml> --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct --run_id my_run
```

Serving a checkpoint for evaluation:

```bash
python deployment/model_server/server_policy.py --ckpt_path <run>/checkpoints/steps_N_pytorch_model.pt --port 10093 --use_bf16
```

## Architecture

**Framework registry.** `starVLA/model/framework/{VLM4A,WM4A,VM4A}/*.py` each define one complete model (VLM-for-action, world-model-for-action, vision-only policies) registered via `@FRAMEWORK_REGISTRY.register("Name")` (`starVLA/model/tools.py`). `build_framework(cfg)` in `starVLA/model/framework/base_framework.py` auto-imports every framework module and instantiates `cfg.framework.name`. A framework file is meant to be the single, self-contained API surface of a model (structurally mirroring its paper diagram); it composes reusable pieces from `starVLA/model/modules/` (`vlm/`, `action_model/`, `world_model/`, `projector/`, `dino_model/`).

**Framework contract.** Frameworks subclass `baseframework` (an HF `PreTrainedModel`) and implement `forward(examples) -> {"action_loss": ...}` and `predict_action(examples) -> {"normalized_actions": ...}`. Both take a **list of raw, model-agnostic dicts** (`image: list[PIL.Image]`, `lang: str`, `action: np.ndarray[T, action_dim]`, optional `state`). All tokenization/image preprocessing happens inside the framework, never in the dataloader. Unnormalization of actions uses `norm_stats` from `dataset_statistics.json` (`FrameworkTools.unnormalize_actions`).

**VLM selection is by path substring.** `starVLA/model/modules/vlm/__init__.py:get_vlm_model` chooses the backbone wrapper by matching `framework.qwenvl.base_vlm` against strings like `Qwen2.5-VL`, `Qwen3-VL`, `Qwen3.5`, `gemma-4`, `florence`, `minicpm-v`, etc. Renaming a local model directory can silently change (or break) which wrapper loads. The config key is `qwenvl` even for non-Qwen backbones.

**Config system.** One global OmegaConf object (YAML + CLI dotlist merge) with top-level `framework`, `datasets` (`vla_data`, `vlm_data`), `trainer`. Each framework declares a `@dataclass <Name>DefaultConfig` merged under `cfg.framework` by `share_tools.merge_framework_config` (YAML overrides defaults; unknown YAML keys are preserved). `share_tools.apply_config_compat` normalizes legacy YAMLs into the current `version_id: "0.21"` schema (e.g. `action_horizon` ↔ `future_action_window_size = action_horizon - 1`) — action heads rely on this having run, so put key-migration logic there rather than in model code.

**Data.** `starVLA/dataloader/__init__.py:build_dataloader` dispatches on `dataset_py` (`lerobot_datasets`, `vlm_datasets`, `umi_datasets`). The LeRobot path (`dataloader/gr00t_lerobot/`) is driven by per-embodiment `DataConfig` classes (video/state/action keys, indices, transforms) and named mixtures. `gr00t_lerobot/registry.py` **auto-discovers** every `examples/**/train_files/data_registry/data_config.py` and merges its `ROBOT_TYPE_CONFIG_MAP` / `DATASET_NAMED_MIXTURES` into the global registries — so a new dataset/benchmark registers itself by adding files under its own `examples/` folder, without editing `starVLA/dataloader/`. `datasets.vla_data.data_mix` selects a mixture by name. See `starVLA/dataloader/gr00t_lerobot/README.md` and `docs/agent_skills/integrate-starvla-dataset/`.

**Training.** `starVLA/training/train_starvla.py` (VLA only), `train_starvla_cotrain.py` (VLA + VLM multimodal co-training, weighted by `trainer.loss_scale`), `train_starvlm.py` (VLM only), `train_starvln.py`. Each is a plain PyTorch/Accelerate loop; per-module learning rates come from `trainer.learning_rate.<top_level_module>` and freezing from `trainer.freeze_modules` (comma-separated module prefixes). Convention: a new training strategy gets its own trainer file rather than branching an existing one.

**Checkpoint layout.** Outputs go to `<run_root_dir>/<run_id>/` containing `config.yaml`, `dataset_statistics.json`, and `checkpoints/steps_N_pytorch_model.pt`. `baseframework.from_pretrained(ckpt)` locates the run dir as `ckpt.parents[1]` and requires both JSON/YAML files there — keep that layout when moving checkpoints.

**Evaluation / deployment.** Evaluation is client–server: `deployment/model_server/server_policy.py` loads a checkpoint and serves `predict_action` over WebSocket (`server_policy_gr00t_zmq.py` speaks the Isaac-GR00T ZMQ protocol). Benchmark/robot clients live in `examples/<category>/<Benchmark>/eval_files/` (e.g. `model2libero_interface.py` wraps `WebsocketClientPolicy`) and typically run in a **separate conda env** for the simulator. Protocol details: `examples/eval_protocol.md`. Server-side config overrides use repeatable `--config_override key=value`.

**Examples layout.** `examples/{simBenchmarks,realRobots,modelExtensions,human2robots}/<Name>/` each hold a README, `train_files/` (YAML, launch script, `data_registry/`) and `eval_files/`. New frameworks/benchmarks should ship an example config here.

## Conventions

- Local assets conventionally live under `playground/` (`Pretrained_models/`, `Datasets/`, `Checkpoints/`), which is not in git. Any `**/bar/` directory is git-ignored — use it for personal scratch scripts.
- Keep changes isolated: avoid modifying shared modules (`starVLA/dataloader/`, `starVLA/training/`, `starVLA/config/`) unless necessary; prefer adding a new framework file or an `examples/` entry.
- Branches: `starVLA` is stable; `starVLA_dev` is the integration branch and the PR target. Commit messages use Conventional Commits (`feat(dataloader): ...`); PR titles use `[type] (scope): description`. Framework/benchmark PRs are expected to include benchmark results and a public checkpoint (see `.github/PULL_REQUEST_TEMPLATE.md`, `docs/PR_readme.md`).
- The repo moves fast; released HF checkpoints may need the code revision from their release time to reproduce exactly.
