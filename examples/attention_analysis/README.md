# LIBERO PI / GR00T attention

2026-10-27：用户已恢复 GPU 权限，并明确取消花屏修复及旧数据重新采集。
80 条评测已完成：PI 40/40、GR00T 37/40 成功；共 77 成功、3 正常任务失败、0 程序错误、0 缺失。
1566 份 attention NPZ、12260 个执行步、160 段视频通过数值、文件、逐动作对齐及解码/帧数验收。
**原 60 段 rollout 的 8k+1 周期性花屏按用户要求保留；没有将其改记为视觉通过。**
正式 summary 的 `evaluation_complete=true` 表示 80 条评测完成；完整报告 `structural_validation_passed=true`，
`accepted=false` 仍保留已知视觉质量门槛。[CPU 诊断](../../playground/attention_analysis/runs/diagnostics/video_flicker/REPORT.md) 保留了历史证据。
[代码审查记录](../../playground/attention_analysis/runs/diagnostics/video_flicker/CODE_REVIEW.md) 记录了候选保护及取证方式；不作旧花屏根因已确认或已修复的结论。
[正式交付索引](../../playground/attention_analysis/deliverables/INDEX.md) · [Codex 续跑指南](../../guideline-for-codex.md)

[全部 80 条验收报告](../../playground/attention_analysis/runs/reports/alltasks_seed7_completed_resume1/summary.json) ·
[容量](../../playground/attention_analysis/runs/reports/alltasks_seed7_completed_resume1/storage.json) ·
[本次日志](../../playground/attention_analysis/runs/tmux/alltasks_seed7_libero10_resume1.log)

## 正式交付结构

```text
playground/attention_analysis/deliverables/
  INDEX.md
  summary.json
  episodes.csv
  libero-spatial-pi/
    INDEX.md
    episodes.json
    task00_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate/
      INDEX.md
      episode.json
      alignment.jsonl
      rollout.mp4
      attention.mp4
      render.json
      overlays/*.png
      traces/<trace_id>/{attention.npz,metadata.json,camera_*.png}
  libero-spatial-groot/
  libero-object-pi/
  libero-object-groot/
  libero-goal-pi/
  libero-goal-groot/
  libero-10-pi/
  libero-10-groot/
```

suite 为 spatial、object、goal、10，modeltype 为 pi、groot；只发布实际完成的任务。
任务内含真实可读取文件，引用使用相对路径；二进制文件可通过硬链接与原始 runs 共享存储。
不发布 smoke、合成验证、部分失败重试或历史候选报告。GR00T goal task06 的正常任务失败保留。

原始实验与排错日志仍在 runs/libero_*，临时叠图/视频在 runs/rendered/，
控制 manifest、完整验收和测试记录在 runs/reports/，环境、模型 revision 与图形库记录在 runs/reproducibility/。

## 数值与图像含义

recorder 默认关闭，仅观察 action expert 实际 cross-attention SDPA 的 Q/K，不替换动作输出路径。
FP32 计算 softmax(scale * QK^T + mask)，先对完整 key 序列归一化，再截取各相机图像 token；保留图像概率质量。
处理 GQA、布尔/additive/causal mask，对 head 求均值；仅记录末尾 action_horizon 个动作 query。
PI 使用逐层 VLM hidden states，导出 36 层；GR00T 使用 VLM 最后一层，仅导出 8 个实际 cross-attention 层。
每相机 NPZ 维度为 [denoise_step, layer, action_query, grid_h, grid_w]；当前为 4 个去噪步、8 个动作 query、8x8 网格。

空间映射由实际 input_ids、image_grid_thw 和 spatial_merge_size 建立，支持非方形图片。
默认展示平均去噪步、层、动作 query，双线性上采样至输入图片，以 50% 透明度叠加。
同一 episode 所有帧及两相机共用色标；chunk 缓存期间显示来源观测热图。
视频中 chunk query 标签用于执行对齐，默认画面仍是所有动作 query 的平均。
primary/wrist 是同一时刻两个相机；当前模型推理没有历史帧输入或跨请求视觉记忆。
热图表示 attention 分配，不是 patch 的因果贡献。

服务端参数为 --attention-config，JSON 配置包含 output_dir、layers、denoise_steps、action_queries、interval。
后三个选择列表默认 null（全部），interval=1 每次推理采集；要求 batch_size=1、diffusion_samples=1。
trace_context 记录 run、task_suite、task_id、episode_idx、env_step、camera_names 和图像变换。
服务端 --seed 设置 Python、NumPy、Torch CPU/CUDA 种子；同一 suite 内 RNG 按默认任务顺序连续推进。

## 验收与发布

使用固定 conda /mnt/afs/wangyunxiangt/miniconda3；所有缓存和临时文件定向到任务目录。
以下命令仅做 CPU 验收与文件整理：

```bash
source examples/attention_analysis/env.sh
export CUDA_VISIBLE_DEVICES=''
"$ATTENTION_PYTHON" -m examples.attention_analysis.suite_report \
  "$ATTENTION_ROOT/runs/reports/libero_current_results"

# 发布一个完整验收通过的报告；支持复用相同来源的已发布任务，拒绝覆盖不同任务结果。
# 当前原 60 条报告因已知视频花屏 accepted=false；保留原交付，不将其重新发布为视觉合格。
"$ATTENTION_PYTHON" -m examples.attention_analysis.publish_results \
  "$ATTENTION_ROOT/runs/reports/libero_current_results"
```

发布器只接受 seed=7、每任务一次、每套完整十任务且验收通过的结果，正常任务失败也发布。
套内的全部结果必须一起保留，不按成功与否筛选。正式交付本身的 summary.json 记录 80 条目标和实际完成数。

```bash
# 使用新标签，完整补跑两模型的最后一套。
bash examples/attention_analysis/launch_all_tasks.sh NEW_LIBERO10_LABEL libero_10 --serial-cross-render

# 新增的完整两套先独立验收并发布到既有交付结构。
"$ATTENTION_PYTHON" -m examples.attention_analysis.suite_report \
  "$ATTENTION_ROOT/runs/reports/NEW_LIBERO10_LABEL"
"$ATTENTION_PYTHON" -m examples.attention_analysis.publish_results \
  "$ATTENTION_ROOT/runs/reports/NEW_LIBERO10_LABEL"

# 组装全部 80 条，保留旧 60 条的视觉质量失败标记。
# 若数值/文件检查通过但 accepted=false，应查看 visual_quality，不能将已知花屏改记为通过。
"$ATTENTION_PYTHON" -m examples.attention_analysis.assemble_report \
  --sources "$ATTENTION_ROOT/runs/reports/libero_current_results" \
            "$ATTENTION_ROOT/runs/reports/NEW_LIBERO10_LABEL" \
  --output "$ATTENTION_ROOT/runs/reports/NEW_COMPLETED_REPORT"
```

serial cross-render 顺序运行两模型、用另一张 GPU 做 EGL 渲染。本次 `alltasks_seed7_libero10_resume1`
已完成两套全部 20 条：PI 10/10、GR00T 8/10；episode 耗时约 7–24 秒，包含正常失败。
续跑、整套组装、CPU 测试、驱动匹配及尚未解决的问题以根目录 guideline-for-codex.md 为准。

`run_rollouts --frame-debug` 会把阶段 PNG、events.jsonl 和 summary.json 写到每个 job 的 frame_debug/，默认关闭。
可单独传 eval 的 `--args.frame-debug-dir`。这些诊断保留在 runs，不混入正式任务交付。
EGL 逐帧绑定默认开启；`STARVLA_EGL_RENDER_GUARD=0` 仅用于恢复授权后的新进程对照，不同时关闭析构修复。
相等的帧 hash 只证明未被改写，不证明渲染内容正常。
