"""Download pinned public weights and assets into the independent Plus root."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

MODEL_REPO = "StarVLA/VLAct_Qwen3PI_Libero_Plus_Finetune"
MODEL_REVISION = "2e288218ddcda26b16c5306f49b9e93799926cea"
MODEL_SHA256 = "16cc54b875d189491f3bd27a41722b82320cfeb21d25a476dd4dc812303d9e7d"
BENCHMARK_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
VLACT_REVISION = "4c41b666228b70d8ae166bb65ac8f6b392ea7919"
BASE_REPO = "StarVLA/Qwen3-VL-4B-Instruct-Action"
BASE_REVISION = "c41500cf1d287cd79e9f6602f419937b05048bd8"
EXPERT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ASSET_REPO = "Sylvest/LIBERO-plus"
ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_benchmark(checkout):
    """Use an exact local copy to avoid repeated AFS reads of all language BDDLs."""
    runtime = Path("/tmp") / f"libero-plus-code-{os.getuid()}-{BENCHMARK_REVISION}"
    marker = runtime / ".plus-revision"
    if not marker.exists():
        runtime.mkdir(exist_ok=True)
        archive = runtime / "source.tar"
        subprocess.run(
            ["git", "-C", str(checkout), "archive", "--format=tar", "--output", str(archive), BENCHMARK_REVISION],
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive), "-C", str(runtime)], check=True)
        marker.write_text(BENCHMARK_REVISION)
    if marker.read_text() != BENCHMARK_REVISION:
        raise ValueError("Local benchmark runtime revision mismatch")
    return runtime


def main():
    import yaml

    from examples.attention_analysis.prepare import download

    root = Path(os.environ["PLUS_ROOT"]).resolve()
    model = root / "checkpoints/libero_plus_pi"
    base = root / "base_models" / BASE_REPO.split("/")[-1]
    expert = root / "base_models/Qwen3-0.6B"
    tasks = []
    for filename in ("config.yaml", "dataset_statistics.json", "README.md"):
        download(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}", model / filename)
    original = model / "config.original.yaml"
    if not original.exists():
        original.write_bytes((model / "config.yaml").read_bytes())
    config = yaml.safe_load(original.read_text())
    config["framework"]["qwenvl"]["base_vlm"] = str(base)
    config["framework"]["action_model"]["expert_model_path"] = str(expert)
    (model / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    checkpoint = model / "checkpoints/steps_50000_pytorch_model.pt"
    tasks.append(
        (f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/checkpoints/{checkpoint.name}", checkpoint)
    )
    revisions = [(BASE_REPO, BASE_REVISION, base)]
    revisions.append(("Qwen/Qwen3-0.6B", EXPERT_REVISION, expert))
    for repo, revision, folder in revisions:
        metadata = root / "cache" / f"{folder.name}-api.json"
        download(f"https://huggingface.co/api/models/{repo}/revision/{revision}", metadata)
        for entry in json.loads(metadata.read_text())["siblings"]:
            name = entry["rfilename"]
            if name.endswith((".json", ".safetensors", ".txt", ".jinja", ".model")):
                tasks.append((f"https://huggingface.co/{repo}/resolve/{revision}/{name}", folder / name))
    archive = root / "cache/assets.zip"
    tasks.append((f"https://huggingface.co/datasets/{ASSET_REPO}/resolve/{ASSET_REVISION}/assets.zip", archive))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, url, path): path for url, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            print(f"Downloaded: {futures[future]}", flush=True)
    print("Verifying checkpoint SHA256", flush=True)
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the published 50K weight")
    benchmark = stage_benchmark(root / "third_party/LIBERO-plus") / "libero/libero"
    # The release ZIP has 457k members beneath a long packaging prefix.
    # Local runtime extraction avoids hundreds of thousands of AFS metadata RPCs;
    # the pinned source archive remains in the independent experiment directory.
    runtime = Path("/tmp") / f"libero-plus-assets-{os.getuid()}-{ASSET_REVISION}"
    runtime.mkdir(exist_ok=True)
    marker = runtime / "extracted.json"
    if not marker.exists():
        print(f"Extracting official asset tree to {runtime}", flush=True)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                _, separator, relative = member.filename.partition("/assets/")
                if not separator:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                target = (runtime / "assets" / relative).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as stream, target.open("wb") as destination:
                        shutil.copyfileobj(stream, destination)
        marker.write_text(json.dumps({"revision": ASSET_REVISION, "archive": str(archive)}))
    link = benchmark / "assets"
    if not link.exists():
        link.symlink_to(runtime / "assets", target_is_directory=True)
    paths = {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "assets": str(runtime / "assets"),
        "datasets": str(root / "datasets"),
    }
    (root / "cache/libero/config.yaml").write_text(yaml.safe_dump(paths))
    for name, revision in (("VLAct", VLACT_REVISION), ("LIBERO-plus", BENCHMARK_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(root / "third_party" / name), "rev-parse", "HEAD"], text=True)
        if actual.strip() != revision:
            raise ValueError(f"Unexpected {name} revision: {actual}")
    manifest = {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODEL_SHA256,
        "training_steps": 50000,
        "benchmark_revision": BENCHMARK_REVISION,
        "vlact_revision": VLACT_REVISION,
        "bases": [{"repo": repo, "revision": rev, "path": str(path)} for repo, rev, path in revisions],
        "asset_revision": ASSET_REVISION,
    }
    (model / "revision.json").write_text(json.dumps(manifest, indent=2))
    (root / "runs/reproducibility/inputs.json").write_text(json.dumps(manifest, indent=2))
    print("Pinned inputs ready", flush=True)


if __name__ == "__main__":
    main()
