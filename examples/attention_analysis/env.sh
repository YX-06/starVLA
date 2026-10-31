#!/usr/bin/env bash
# Source from any directory. All downloads and configurable caches stay on AFS.
ATTENTION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ATTENTION_ROOT="$ATTENTION_REPO/playground/attention_analysis"
mkdir -p "$ATTENTION_ROOT"/{checkpoints,base_models,datasets,third_party,cache,tmp,runs,deliverables}
export CONDA_EXE=/mnt/afs/wangyunxiangt/miniconda3/bin/conda
export ATTENTION_PYTHON=/mnt/afs/wangyunxiangt/miniconda3/envs/starvla-attn/bin/python
export HF_HOME="$ATTENTION_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ATTENTION_ROOT/cache/torch"
export PIP_CACHE_DIR="$ATTENTION_ROOT/cache/pip"
export PIP_NO_COMPILE=1
export CONDA_PKGS_DIRS="$ATTENTION_ROOT/cache/conda"
export TMPDIR="$ATTENTION_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$ATTENTION_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$ATTENTION_ROOT/cache/xdg"
export TRITON_CACHE_DIR="$ATTENTION_ROOT/cache/triton"
export NUMBA_CACHE_DIR="$ATTENTION_ROOT/cache/numba"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ATTENTION_REPO:$ATTENTION_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export __GL_SHADER_DISK_CACHE_PATH="$ATTENTION_ROOT/cache/nvidia"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$ATTENTION_ROOT/cache/libero"
export NO_ALBUMENTATIONS_UPDATE=1
# LIBERO's pinned initial-state files contain NumPy arrays serialized by torch.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ -f "$ATTENTION_ROOT/cache/nvidia/10_nvidia.json" ]]; then
    export LD_LIBRARY_PATH="$ATTENTION_ROOT/third_party/nvidia-550.90.07:$ATTENTION_ROOT/cache/graphics-loader${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$ATTENTION_ROOT/cache/nvidia/10_nvidia.json"
fi
