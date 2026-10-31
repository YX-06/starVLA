# LIBERO-Plus PI attention

This workflow evaluates a **280-task stratified subset**, not all 10,030 Plus
tasks. Each original LIBERO task contributes one SHA256-selected variant from
each of seven perturbation classes. The frozen JSON records the selection hash,
candidate count, official ID, runtime index, category and difficulty. Execution
order is spatial, object, goal, 10, with ascending runtime index within each suite.

Inputs are pinned in `prepare.py`: the public 50K
[VLAct checkpoint](https://huggingface.co/StarVLA/VLAct_Qwen3PI_Libero_Plus_Finetune/tree/2e288218ddcda26b16c5306f49b9e93799926cea),
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182),
VLAct code, Action backbone and assets. The expert initialization dependency is
pinned to its recorded revision. Checkpoint SHA256 must match
the publisher's value. This policy was fine-tuned on original LIBERO and is
evaluated zero-shot on Plus.

All new artifacts live under `playground/attention_analysis_plus/`. The original
LIBERO artifacts and environments remain available. The isolated Python virtual
environments inherit the existing `starvla-attn`/`libero-attn` dependencies without
modifying those environments. Added dependencies install only in the new venvs.
The official ZIP contains a long packaging prefix and 457k members; the loader
strips that prefix and extracts runtime assets on `/tmp`, retaining the pinned
archive on AFS. Re-run preparation after a host change to rebuild that cache.

```bash
cd /mnt/afs/wangyunxiangt/workspace/starVLA
source examples/attention_analysis_plus/env.sh

# Once the two independent environments and local ImageMagick libraries exist:
"$PLUS_PYTHON" -u -m examples.attention_analysis_plus.prepare
"$PLUS_PYTHON" -m examples.attention_analysis_plus.tasks
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx CUDA_VISIBLE_DEVICES='' \
  "$PLUS_SIM_PYTHON" -m examples.attention_analysis_plus.preflight

CUDA_VISIBLE_DEVICES=0 "$PLUS_PYTHON" -m examples.attention_analysis_plus.smoke \
  --checkpoint "$PLUS_ROOT/checkpoints/libero_plus_pi/checkpoints/steps_50000_pytorch_model.pt" \
  --output "$PLUS_ROOT/runs/model_parity_repro"
cp "$PLUS_ROOT/runs/model_parity_repro/verification.json" \
  "$PLUS_ROOT/runs/reproducibility/model-parity.json"
bash examples/attention_analysis_plus/launch.sh smoke_seed7_r1 --smoke
# After the complete smoke episode passes and its cost is reviewed:
bash examples/attention_analysis_plus/launch.sh plus_seed7_r1

tmux -S /tmp/starvla-plus-$(id -u).sock attach -t plus_seed7_r1
tail -f "$PLUS_ROOT/runs/tmux/plus_seed7_r1.log"
```

Existing run/manifest paths are refused, preserving failed attempts. Each suite
starts a fresh seed-7 model server on GPU 0. Within that suite the server RNG
advances continuously. A fresh simulator process on GPU 1 runs each task at
seed 7, `initial_states[0]`, with ten stabilization steps and the official
220/280/300/520 action-step limits. Program errors are recorded and later tasks
continue. A program error is distinct from reaching the normal task step limit.

Model input processing uses the pinned VLAct framework unchanged: primary then
wrist, 180-degree rotation, client resize to 224, published CoT prompt, and no
state input (matching the released evaluation client). Action unnormalization
matches the **published Plus client code**, which uses `min/max/mask` and a 0.5
gripper threshold. The model card mentions `q01/q99`, but that is different from
the actual released Plus client; the implementation and metadata state the
formula used explicitly. No action ensembling is applied by that client.

The recorder observes only the actual SDPA inside the 28 expert cross-attention
modules. It computes FP32 probabilities with the actual Q/K over the **complete
prefix**, then extracts the two image spans and averages heads. Every request
saves all 4 denoising steps and 8 action queries, shape
`[4, 28, 8, grid_h, grid_w]` per camera. No image-only renormalization is applied.

The report checks every NPZ, task identity, initial state/seed metadata, finite
nonnegative probabilities, combined camera probability mass, input image
decodability, normalized/unnormalized/executed action alignment, and complete
video decoding. Videos preserve all original frames; isolated-spike diagnostics
are recorded for **both cameras**. No corruption repair or frame postprocessing
is performed, and the heuristic is not a guarantee of visual correctness.

An episode uses one probability color scale across all requests and both
cameras. Display maps average layers, denoising steps and action queries; query
labels identify the executed action. Attention is not causal attribution.

```text
deliverables/libero-plus-<suite>-pi/task<official_id>_<variant_name>/
  rollout.mp4, rollout_wrist.mp4, attention.mp4
  episode.json, alignment.jsonl, render.json, validation.json
  overlays/*_camera_{0,1}.png, *_combined.png
  chunks/<trace_id>.npz             # normalized and physical action chunks
  traces/<trace_id>/
    attention.npz, metadata.json, camera_0.png, camera_1.png
```

`summary.json`, `episodes.csv`, `task_manifest.json` and `INDEX.md` form the final
delivery entry points. Success denominators are fixed at 70 per suite, 40 per
perturbation class and 280 overall. Missing/program-error episodes prevent final
publication. Normal task failures are preserved. Smoke artifacts remain in runs.
Logical artifact bytes, per-request bytes and elapsed seconds are recorded;
hardlinks share the original numerical/video bytes with the delivery directory.

```bash
export ATTENTION_ROOT="$PLUS_REPO/playground/attention_analysis"
CUDA_VISIBLE_DEVICES='' "$PLUS_PYTHON" -m pytest \
  examples/attention_analysis_plus examples/attention_analysis \
  --ignore=examples/attention_analysis/test_eval_records.py -q -p no:cacheprovider
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx CUDA_VISIBLE_DEVICES='' \
  "$PLUS_SIM_PYTHON" -m pytest examples/attention_analysis/test_eval_records.py -q -p no:cacheprovider
"$PLUS_PYTHON" -m black --check examples/attention_analysis_plus starVLA/model/analysis/attention.py
"$PLUS_PYTHON" -m ruff check examples/attention_analysis_plus starVLA/model/analysis/attention.py
```
