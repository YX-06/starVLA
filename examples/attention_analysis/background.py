"""Durable tmux orchestration; every stage writes its exit code and log path."""

import concurrent.futures
import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from examples.attention_analysis.prepare import MODELS

ROOT = Path(os.environ["ATTENTION_ROOT"])
CONDA = Path("/mnt/afs/wangyunxiangt/miniconda3")
PYTHON = str(CONDA / "envs/starvla-attn/bin/python")
LOGS = ROOT / "runs/tmux"


def stage(name, command, *, env=None, retries=1):
    path = LOGS / f"{name}.log"
    previous = LOGS / f"{name}.json"
    if previous.exists() and json.loads(previous.read_text()).get("exit_code") == 0:
        print(f"[{time.strftime('%F %T')}] REUSE passed stage {name}", flush=True)
        return True
    result = None
    for attempt in range(retries):
        print(f"[{time.strftime('%F %T')}] START {name} attempt={attempt + 1} log={path}", flush=True)
        with path.open("a") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        status = {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "attempt": attempt + 1,
            "log": str(path),
            "finished_at": time.strftime("%F %T"),
        }
        (LOGS / f"{name}.json").write_text(json.dumps(status, indent=2))
        print(f"[{time.strftime('%F %T')}] END {name} exit={result.returncode}", flush=True)
        if result.returncode == 0:
            return True
        if attempt + 1 < retries:
            time.sleep(5)
    return False


def install_simulators():
    dependencies = {
        "libero-attn": [
            "torch==2.6.0",
            "torchvision==0.21.0",
            "numpy==1.24.4",
            "mujoco==3.2.3",
            "robosuite==1.4.0",
            "robomimic==0.2.0",
            "bddl==1.0.1",
            "gym==0.25.2",
            "hydra-core==1.3.2",
            "easydict",
            "future",
            "cloudpickle",
            "thop",
            "einops",
            "tyro",
            "matplotlib",
            "imageio",
            "imageio-ffmpeg",
            "tqdm",
            "websockets",
            "websocket-client",
            "msgpack",
            "opencv-python-headless",
        ],
    }
    outcomes = {}
    for name, packages in dependencies.items():
        python = str(CONDA / "envs" / name / "bin/python")
        outcomes[name] = stage(f"install-{name}", [python, "-m", "pip", "install", "--no-compile", *packages], retries=2)
        if name == "libero-attn" and outcomes[name]:
            outcomes[name] = stage(
                "install-libero-source",
                [python, "-m", "pip", "install", "--no-deps", "--no-compile", "-e", str(ROOT / "third_party/LIBERO")],
            )
    return outcomes


def prepare_libero_config():
    source = ROOT / "third_party/LIBERO/libero/libero"
    directory = Path(os.environ["LIBERO_CONFIG_PATH"])
    directory.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; avoid LIBERO's interactive first-import prompt.
    config = {
        "benchmark_root": str(source),
        "bddl_files": str(source / "bddl_files"),
        "init_states": str(source / "init_files"),
        "assets": str(source / "assets"),
        "datasets": str(ROOT / "datasets/libero"),
    }
    (ROOT / "datasets/libero").mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(json.dumps(config, indent=2))


def export_environments():
    output = ROOT / "runs/reproducibility/environments"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("starvla-attn", "libero-attn"):
        python = str(CONDA / "envs" / name / "bin/python")
        with (output / f"{name}-pip.txt").open("w") as stream:
            subprocess.run([python, "-m", "pip", "freeze"], stdout=stream, check=False)
        with (output / f"{name}-conda.yaml").open("w") as stream:
            subprocess.run([str(CONDA / "bin/conda"), "env", "export", "-n", name], stdout=stream, check=False)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/reproducibility").mkdir(parents=True, exist_ok=True)
    lock = (Path("/tmp") / f"starvla-attn-pipeline-{os.getuid()}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"hostname": socket.gethostname(), "pid": os.getpid(), "status": "running", "stages": {}}
    state_file = LOGS / "pipeline.json"
    state_file.write_text(json.dumps(status, indent=2))
    print(f"Pipeline pid={os.getpid()} host={socket.gethostname()} root={ROOT}", flush=True)
    try:
        prepare_libero_config()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            simulators = pool.submit(install_simulators)
            status["stages"]["unit-tests"] = stage(
                "unit-tests", [PYTHON, "-m", "pytest", "examples/attention_analysis", "-q"]
            )
            for benchmark in ("libero",):
                ready = stage(
                    "download-libero",
                    [PYTHON, "-m", "examples.attention_analysis.prepare", "--weights"],
                    retries=3,
                )
                status["stages"][f"download-{benchmark}"] = ready
                state_file.write_text(json.dumps(status, indent=2))
                if not ready:
                    continue
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
                for index, model in enumerate(("pi", "groot")):
                    name = f"{benchmark}_{model}"
                    step = MODELS[name][1]
                    checkpoint = ROOT / "checkpoints" / name / "checkpoints" / f"steps_{step}_pytorch_model.pt"
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[index])
                    status["stages"][f"smoke-{name}"] = stage(
                        f"smoke-{name}",
                        [
                            PYTHON,
                            "-m",
                            "examples.attention_analysis.smoke",
                            "--checkpoint",
                            str(checkpoint),
                            "--cameras",
                            "2",
                            "--output",
                            str(ROOT / "runs" / f"{name}_smoke"),
                        ],
                        env=env,
                    )
                    state_file.write_text(json.dumps(status, indent=2))
                simulator_ready = simulators.result()
                status["stages"].update({f"install-{name}": ready for name, ready in simulator_ready.items()})
                if simulator_ready[f"{benchmark}-attn"]:
                    status["stages"][f"rollout-{benchmark}"] = stage(
                        f"rollout-{benchmark}-final",
                        [
                            PYTHON,
                            "-m",
                            "examples.attention_analysis.run_rollouts",
                            "--benchmark",
                            benchmark,
                            "--episodes",
                            "1",
                            "--run-label",
                            "final",
                        ],
                    )
                    state_file.write_text(json.dumps(status, indent=2))
        stage(
            "synthetic-numerical-export",
            [
                PYTHON,
                "-m",
                "examples.attention_analysis.test_attention",
                "--output",
                str(ROOT / "runs/diagnostics/synthetic_validation"),
            ],
        )
        stage(
            "synthetic-render",
            [
                PYTHON,
                "-m",
                "examples.attention_analysis.render",
                str(ROOT / "runs/diagnostics/synthetic_validation/traces"),
                "--output",
                str(ROOT / "runs/diagnostics/synthetic_validation/rendered"),
            ],
        )
        manifests = [json.loads(path.read_text()) for path in (ROOT / "checkpoints").glob("*/revision.json")]
        (ROOT / "runs/reproducibility/model_manifest.json").write_text(json.dumps(manifests, indent=2))
        export_environments()
        status["status"] = "finished" if all(status["stages"].values()) else "finished_with_failures"
    except Exception as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        state_file.write_text(json.dumps(status, indent=2))
        (ROOT / "runs/reproducibility/pipeline_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
