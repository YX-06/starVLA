"""Extract version-matched NVIDIA user-space graphics libraries into the workspace.

This never invokes the NVIDIA installer or changes a kernel driver. The EGL
loader JSON points directly at the extracted libraries.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from examples.attention_analysis.prepare import download


def main():
    root = Path(os.environ["ATTENTION_ROOT"])
    version = "550.90.07"
    actual = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    if set(actual) != {version}:
        raise RuntimeError(f"Expected driver {version}, found {actual}")
    name = f"NVIDIA-Linux-x86_64-{version}-no-compat32.run"
    archive = root / "third_party" / name
    checksum = root / "third_party" / f"{name}.sha256sum"
    base = f"https://download.nvidia.com/XFree86/Linux-x86_64/{version}"
    download(f"{base}/{name}", archive)
    download(f"{base}/{name}.sha256sum", checksum)
    expected = checksum.read_text().split()[0]
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("NVIDIA graphics archive checksum mismatch")
    target = root / "third_party" / f"nvidia-{version}"
    if not (target / f"libEGL_nvidia.so.{version}").exists():
        subprocess.run(["sh", str(archive), "--extract-only", "--target", str(target)], check=True)
    for library in ("libEGL_nvidia", "libGLX_nvidia"):
        link = target / f"{library}.so.0"
        if not link.exists():
            link.symlink_to(f"{library}.so.{version}")
    cache = root / "cache/nvidia"
    cache.mkdir(parents=True, exist_ok=True)
    loader = root / "cache/graphics-loader"
    loader.mkdir(parents=True, exist_ok=True)
    conda_lib = Path("/mnt/afs/wangyunxiangt/miniconda3/envs/libero-attn/lib")
    for pattern in ("libEGL.so*", "libOpenGL.so*", "libGL.so*", "libGLdispatch.so*", "libGLX.so*"):
        for library in conda_lib.glob(pattern):
            link = loader / library.name
            if not link.exists():
                link.symlink_to(library.resolve())
    (cache / "10_nvidia.json").write_text(
        json.dumps(
            {"file_format_version": "1.0.0", "ICD": {"library_path": str(target / "libEGL_nvidia.so.0")}}, indent=2
        )
    )
    output = root / "runs/reproducibility"
    output.mkdir(parents=True, exist_ok=True)
    (output / "graphics.json").write_text(
        json.dumps(
            {
                "driver_version": version,
                "archive_url": f"{base}/{name}",
                "archive_sha256": expected,
                "library_directory": str(target),
                "kernel_driver_modified": False,
            },
            indent=2,
        )
    )
    print(f"Graphics libraries ready: {target}")


if __name__ == "__main__":
    main()
