#!/usr/bin/env bash
# Source from the repository root; legacy LIBERO paths remain independent.
PLUS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PLUS_ROOT="$PLUS_REPO/playground/attention_analysis_plus"
export PLUS_PYTHON="$PLUS_ROOT/envs/vlact/bin/python"
export PLUS_SIM_PYTHON="$PLUS_ROOT/envs/libero-plus/bin/python"
export HF_HOME="$PLUS_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PLUS_ROOT/tmp"
export MPLCONFIGDIR="$PLUS_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$PLUS_ROOT/cache/xdg"
export TORCH_HOME="$PLUS_ROOT/cache/torch"
export PIP_CACHE_DIR="$PLUS_ROOT/cache/pip"
export TRITON_CACHE_DIR="$PLUS_ROOT/cache/triton"
export NUMBA_CACHE_DIR="$PLUS_ROOT/cache/numba"
export LIBERO_CONFIG_PATH="$PLUS_ROOT/cache/libero"
export LIBERO_HOME="$PLUS_ROOT/third_party/LIBERO-plus"
PLUS_LIBERO_RUNTIME="/tmp/libero-plus-code-$(id -u)-4976dc30028e805ff8094b55501d532c48fec182"
if [[ -f "$PLUS_LIBERO_RUNTIME/.plus-revision" ]]; then
    export LIBERO_HOME="$PLUS_LIBERO_RUNTIME"
fi
export PYTHONPATH="$PLUS_REPO:$LIBERO_HOME"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
# Retain the existing destructor fix; do not enable the cancelled frame repair.
export STARVLA_EGL_RENDER_GUARD=0
export LD_LIBRARY_PATH="$PLUS_ROOT/third_party/imagemagick/usr/lib/x86_64-linux-gnu:$PLUS_REPO/playground/attention_analysis/third_party/nvidia-550.90.07:$PLUS_REPO/playground/attention_analysis/cache/graphics-loader${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_CONFIGURE_PATH="$PLUS_ROOT/third_party/imagemagick/etc/ImageMagick-6"
export MAGICK_HOME="$PLUS_ROOT/third_party/imagemagick"
export MAGICK_CODER_MODULE_PATH="$PLUS_ROOT/third_party/imagemagick/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/modules-Q16/coders"
export __EGL_VENDOR_LIBRARY_FILENAMES="$PLUS_REPO/playground/attention_analysis/cache/nvidia/10_nvidia.json"
mkdir -p "$TMPDIR" "$LIBERO_CONFIG_PATH" "$PLUS_ROOT/runs/reproducibility"
