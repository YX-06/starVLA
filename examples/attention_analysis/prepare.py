"""Pin remote revisions and download experiment inputs beneath ATTENTION_ROOT.

Metadata is always downloaded; --weights additionally fetches checkpoints and
their base VLMs. config.original.yaml preserves all published settings; only
local model paths are changed in config.yaml. Never use an unpinned main for
weights after the manifest has been written.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

MODELS = {
    "libero_pi": ("StarVLA/Qwen3-VL-PI-LIBERO-4in1", 100000),
    "libero_groot": ("StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1", 30000),
}


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Locks live on local storage: some AFS mounts return EAGAIN even for
    # blocking flock. Both GPU lanes run on this same host.
    lock_root = Path("/tmp") / f"starvla-attn-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_name = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()
    with (lock_root / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _download_locked(url, destination)


def _download_locked(url, destination):
    if destination.exists() and destination.stat().st_size:
        if not _is_lfs_pointer(destination):
            return
        destination.replace(destination.with_name(destination.name + ".lfs-pointer"))
    temporary = destination.with_name(destination.name + ".partial")
    text_file = destination.suffix in {".json", ".yaml", ".yml", ".txt", ".jinja"}
    endpoints = [url.replace("/resolve/", "/raw/"), url] if text_file else [url]
    for endpoint in dict.fromkeys(endpoints):
        command = [
            "curl",
            "-sSL",
            "--fail",
            "--retry",
            "3",
            "--connect-timeout",
            "10",
            "--max-time",
            "120" if text_file else "7200",
        ]
        if not text_file:
            command += ["-C", "-"]
        command += [endpoint, "-o", str(temporary)]
        for attempt in range(4):
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                break
            if attempt == 3:
                raise subprocess.CalledProcessError(result.returncode, command)
            time.sleep(2)
        if not _is_lfs_pointer(temporary):
            temporary.replace(destination)
            return
    raise ValueError(f"Received an LFS pointer instead of file contents: {url}")


def _is_lfs_pointer(path):
    with path.open("rb") as stream:
        return stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def pin(repo, destination):
    download(f"https://huggingface.co/api/models/{repo}", destination)
    data = json.loads(destination.read_text())
    return data["sha"], [entry["rfilename"] for entry in data["siblings"]]


def prepare(name, root, weights):
    repo, step = MODELS[name]
    revision, files = pin(repo, root / "checkpoints" / f"{name}.json")
    folder = root / "checkpoints" / name
    for filename in ("config.yaml", "dataset_statistics.json"):
        download(f"https://huggingface.co/{repo}/raw/{revision}/{filename}", folder / filename)
    original = folder / "config.original.yaml"
    if not original.exists():
        original.write_text((folder / "config.yaml").read_text())
    config = yaml.safe_load(original.read_text())
    backbone_reference = config["framework"]["qwenvl"]["base_vlm"].rstrip("/")
    backbone = backbone_reference.split("/")[-1]
    backbone_repo = backbone_reference if backbone_reference.startswith(("Qwen/", "StarVLA/")) else f"Qwen/{backbone}"
    base_revision, base_files = pin(backbone_repo, root / "base_models" / f"{backbone}.json")
    base_folder = root / "base_models" / backbone
    config["framework"]["qwenvl"]["base_vlm"] = str(base_folder)
    bases = [(backbone_repo, base_revision, base_files, base_folder)]
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    selected = [filename for filename in files if f"steps_{step}_" in filename and filename.endswith(".pt")]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one checkpoint at step {step}, found {selected}")
    manifest = {
        "repo": repo,
        "revision": revision,
        "checkpoint": selected[0],
        "training_steps": step,
        "bases": [{"repo": repo_id, "revision": rev, "path": str(path)} for repo_id, rev, _, path in bases],
    }
    (folder / "revision.json").write_text(json.dumps(manifest, indent=2))
    if weights:
        download(f"https://huggingface.co/{repo}/resolve/{revision}/{selected[0]}", folder / selected[0])
        for repo_id, rev, filenames, path in bases:
            for filename in sorted(filenames, key=lambda name: not name.endswith(".safetensors")):
                if filename.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")):
                    download(f"https://huggingface.co/{repo_id}/resolve/{rev}/{filename}", path / filename)
        # A validation/repair process may have quarantined a legacy download
        # while base models were being prepared. Recheck before declaring ready.
        download(f"https://huggingface.co/{repo}/resolve/{revision}/{selected[0]}", folder / selected[0])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["ATTENTION_ROOT"])
    # Base folders can be shared by jobs; prepare serially to avoid partial-file races.
    manifests = [prepare(name, root, args.weights) for name in args.models]
    output = root / "runs/reproducibility"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_manifest.json").write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
