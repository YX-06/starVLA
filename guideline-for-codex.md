# LIBERO PI / GR00T：后续 Codex 续跑指南

## 当前状态：评测已完成（2026-10-27）

80/80 条已执行：PI spatial/object/goal/10 均为 10/10，总计 40/40；
GR00T 为 10/10、10/10、9/10、8/10，总计 37/40。
共 77 成功、3 正常失败、0 程序错误、0 缺失。
1566 个 trace、12260 个动作执行步、160 段视频的结构验收通过，77 项测试及 Black/Ruff 通过。

- 正式产物仍在 [deliverables](playground/attention_analysis/deliverables/INDEX.md)，共八个 suite/model、80 个 task 目录。
- [完整报告](playground/attention_analysis/runs/reports/alltasks_seed7_completed_resume1/summary.json)、
  [容量](playground/attention_analysis/runs/reports/alltasks_seed7_completed_resume1/storage.json)。
- 原 60 条来自 `libero_current_results`；新增两套来自 `alltasks_seed7_libero10_resume1`，全部正常失败保留。
- 用户明确取消花屏修复；旧 60 段视频及 `visual_quality=failed` 标记保持原样。
  `evaluation_complete=true` 与 `structural_validation_passed=true` 表示评测/结构完成，
  `accepted=false` 及 `full_evaluation_complete=false` 保留视觉质量门槛，不应据此自动重跑任务。
- 没有剩余评测任务。除非用户另有指示，不再启动续跑、修复花屏或重新采集。

以下为此前交接过程与复现说明；与当前状态冲突时，以本节及用户最新指示为准。

## 2026-10-27 用户最新指示（优先）

GPU 使用权限已明确恢复，当前节点两张 H100/驱动 550.90.07 已核实可用。
用户随后明确表示“不需要修复花屏了”：停止花屏修复、额外取证和旧 60 条的重新采集，
保留既有结果，继续剩余两模型的 `libero_10` 共 20 条。
已知视频质量失败标记和历史失败记录仍保留，不将取消修复表述为视觉验收通过。
本轮完成的运行标签为 `alltasks_seed7_libero10_resume1`，采用 serial cross-render；
实际状态以该运行 status.json、`runs/reports/` 报告及正式交付索引为准。
下文 GPU 暂停及花屏优先修复要求属于此前交接状态，遇冲突时以上述最新用户指示为准。

## 先读状态，暂不使用 GPU

用户暂时被收回 GPU 使用权限。本次交接只整理已有实验、做 CPU 验证和清理冗余。
**必须等用户明确告知权限恢复，才能启动 CUDA 推理、GPU 渲染或 GPU 复现。**
不要因为看到旧 `running`、旧 PID、旧 tmux 命令或本文件中的续跑命令就自动开跑。

- 仓库：`/mnt/afs/wangyunxiangt/workspace/starVLA`。
- 当前权威入口：[交付索引](playground/attention_analysis/deliverables/INDEX.md)。
- 当前机器可读状态：[handoff_state.json](playground/attention_analysis/runs/reports/libero_current_results/handoff_state.json)。
- 正式结果：[summary.json](playground/attention_analysis/deliverables/summary.json)、[逐任务表](playground/attention_analysis/deliverables/episodes.csv)。
- 已完成部分的完整验收：[validation summary](playground/attention_analysis/runs/reports/libero_current_results/summary.json)。
- 旧 smoke、synthetic、retry 和未完成候选报告已从 deliverables 清除，不再以这些目录作为正式交付入口。
- 本次整理所在节点与原 GPU worker 不同；本地没有原 tmux。旧 GPU 节点的进程存活状态未核实，不要按历史 PID 直接发送信号。

## 新发现的阻塞项：原始 rollout 每 8 帧花屏

CPU 全量检查确认全部 60 段 rollout 存在周期性单帧异常，位置为 8k+1，共检出 874 帧；
视频码流可以正常解码，原来的帧数/数值验收没有覆盖视觉内容正确性。
详见 [画面质量诊断](playground/attention_analysis/runs/diagnostics/video_flicker/REPORT.md)。
执行结果仍是 60 条、59 成功和 1 失败，但**原始 rollout 视频质量不合格，交付不能算最终验收通过**。
892 张保存的 primary 推理输入对应 8k 帧，与视频该位置匹配，抽查画面正常；不能据此断言所有相机或其他中间观测均无问题。
花屏位置紧接每次 chunk 推理之后，疑似采集/渲染读回链路，但尚未用原始坏帧 PNG 定位，不能将 EGL/驱动故障写成已证实根因。

恢复 GPU 权限后，先对最小真实 rollout 同时保存每步原始相机 PNG、推理前后缓冲副本和帧校验，
对比编码前 PNG 与解码帧，定位到渲染读回、缓冲管理或编码输入中的具体环节，再修复并验证。
至少覆盖多个完整 8 步 chunk，不能仅检查 8k 时刻的模型输入 PNG。
不要直接继续最后 20 条，也不要通过删帧、重复帧或插值掩盖问题。
旧视频缺少损坏位置的原始无损 PNG，不能承诺通过重新编码恢复；若修复需要重新采集，应按完整 suite/模型重跑，保留原始失败记录，不挑成功 episode。
当前报告显式区分 structural_validation_passed 和 accepted；已知 visual_quality=failed 的 suite 会阻止总验收及新发布。

本轮代码审查已实现候选保护与逐阶段取证，见 [CODE_REVIEW.md](playground/attention_analysis/runs/diagnostics/video_flicker/CODE_REVIEW.md)：

- `render/read_pixels` 每次先绑定 owner EGL context，缺失/释放/绑定失败时提前抛错；仅在 worker 内包装类，不改 conda 安装。
- 录像帧和实际请求图像使用独立副本，缓存动作步不再做无用的图像 resize。
- `run_rollouts --frame-debug` 或 eval `--args.frame-debug-dir` 保存推理前、推理后、env.step 后及编码前的无损 PNG/hash/线程/context。
  前三阶段保存两相机，编码前保存原始录像所用的 primary；诊断默认关闭，不增加 render/step/随机采样。
- 通过 `STARVLA_EGL_RENDER_GUARD=0/1` 在全新进程中 A/B。0 仅关闭新逐帧绑定保护，保留已有析构修复。
  固定值会写入每个 job 的 status.json 并传给 worker。不要用 --legacy-cleanup 同时关闭多项修复。
- CPU 真实 WebSocket、复用/故意修改图像缓冲区的 mock 环境和 context 丢失/绑定失败测试已通过。
  **这些不是实际 GPU 修复验证；旧花屏问题仍未闭环，不能撤掉质量失败标记。**

## 已完成与剩余

协议固定为 seed=7、默认 task_order_index=0、每任务一个 episode、`initial_states[0]`。
模型是现有固定 revision 的 Qwen3-VL PI 100K、Qwen2.5-VL GR00T 30K。

| 模型 | spatial | object | goal | libero_10 | 当前有效结果 |
|---|---:|---:|---:|---|---|
| PI | 10/10 | 10/10 | 10/10 | 未完成 | 30/30 成功，另有 10 个待评测 |
| GR00T | 10/10 | 10/10 | 9/10 | 未完成 | 29/30 成功，另有 10 个待评测 |

共 60 条已执行 episode，59 成功、1 正常任务失败；数值/对齐及文件结构检查通过，但上述原始视频画面质量问题尚未解决。
失败为 GR00T `libero_goal` task 6，`put the cream cheese in the bowl`，达到 300 动作步上限。
保留这条失败，不能挑选成功结果覆盖它。全任务目标是 80 条，剩余两模型各自完整的 `libero_10` 共 20 条。

唯一选用的前三套原始数据：

```text
playground/attention_analysis/runs/libero_{pi,groot}_alltasks_seed7_retry1/
  {libero_spatial,libero_object,libero_goal}/
playground/attention_analysis/deliverables/libero-<suite>-<modeltype>/
  taskNN_<task_name>/
    rollout.mp4
    attention.mp4
    episode.json
    alignment.jsonl
    render.json
    overlays/*.png
    traces/<trace_id>/{attention.npz,metadata.json,camera_*.png}
```

`suite` 为 spatial、object、goal、10；`modeltype` 为 pi、groot。目前只有已完成的六个 suite/model 目录及 60 个 task 目录。
每任务均包含真实数值、输入图片及视频文件，不依赖指向 runs 的符号链接；二进制文件可通过硬链接共享存储。
任务内的文件引用使用相对路径。`runs/reports/libero_current_results` 当前 `structural_validation_passed=true`、`accepted=false`，因视频画面质量不合格。
后续的临时 render 输出位于 `runs/rendered/`，控制 manifest/验收报告位于 `runs/reports/`；
必须用 `publish_results` 将验收通过的整套结果发布到正式 deliverables，不把 smoke 或失败重试报告放回去。

## 保留的实现和诊断

- attention 默认关闭；开启后观察 action expert 实际 SDPA 的 Q/K，FP32 计算完整 key softmax，再截取图片 token，平均 head。
- PI 导出 36 层，GR00T 仅导出偶数编号的 8 个 cross-attention 层；每次推理 4 个去噪步、8 个动作 query、primary/wrist 两相机。每相机当前网格 8×8。
- 默认热图平均去噪步、层、动作 query；视频上的 chunk query 是执行对齐标签，画面默认仍是全部 query 的平均。缓存动作时复用来源观测热图。
- 当前推理只看同一时刻两相机，没有历史帧输入或跨请求视觉 KV 记忆。保持相机顺序、prompt、预处理、state 使用方式及采样逻辑。
- 全任务采用每任务独立 simulator 子进程，同一 suite 内保留同一个 policy server。每个 suite 新 server 设置 seed=7，suite 内 RNG 连续推进；不要逐任务重置服务端 seed 或只补跑个别 task。
- `egl_cleanup.py` 修复 robosuite 延迟析构时绑定错误 GL context；`task_worker.py` 隔离 native abort/timeout，默认每任务 timeout 3600 秒。
- 原多任务 EGL abort 的诊断在 `runs/diagnostics/native_diagnosis.json`；最小复现在 `examples/attention_analysis/repro_multitask.py`，日志在 `runs/tmux/*repro.log`。
- `libero_10` 的同卡推理与 EGL 渲染曾非常慢；无模型渲染正常。旧 600 秒 timeout 重试没有完成整套。
- 最新实现支持 `--serial-cross-render`：PI 推理 GPU 0、渲染 GPU 1；PI 完成后，GR00T 推理 GPU 1、渲染 GPU 0。
  此模式尚无完整真实 episode 验证，不能写成已解决性能问题。
- retry3/retry4 没有完成任何 `libero_10` episode。retry4 的旧状态为 PI `running`、GR00T `ready`，属于中断时留下的快照。
- 失败尝试的原始运行目录、日志、数值和 status.json 保留作诊断；旧重复交付副本已删除，不混入有效 60 条。

## 无 GPU 时可以做的检查

固定使用 `/mnt/afs/wangyunxiangt/miniconda3`，保留 `starvla-attn`、`libero-attn`，不修改 base。
下载、缓存和临时目录均由 `env.sh` 定向到 `playground/attention_analysis`。

```bash
cd /mnt/afs/wangyunxiangt/workspace/starVLA
source examples/attention_analysis/env.sh
export CUDA_VISIBLE_DEVICES=''

# CUDA 数值用例应跳过；其他测试不需要实际 GPU。
"$ATTENTION_PYTHON" -m pytest examples/attention_analysis \
  --ignore=examples/attention_analysis/test_eval_records.py -q -p no:cacheprovider

# 纯 mock 仿真适配器测试；glx 仅避免导入时初始化 EGL，不创建窗口或真实 simulator。
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx \
  /mnt/afs/wangyunxiangt/miniconda3/envs/libero-attn/bin/python \
  -m pytest examples/attention_analysis/test_eval_records.py -q -p no:cacheprovider

# 重新验收已完成的 60 条数据；含完整视频解码，纯 CPU。
# 当前预期 exit code=1：已知视频花屏会阻止 accepted，不代表数值/解码回归失败。
"$ATTENTION_PYTHON" -m examples.attention_analysis.suite_report \
  "$ATTENTION_ROOT/runs/reports/libero_current_results"
```

当前测试、源码和清理记录位于 `runs/reports/libero_current_results/`；环境导出、模型 revision 和图形库记录位于 `runs/reproducibility/`。
历史 GPU 测试证据与本次 CPU 验证分别保存，不把跳过的 CUDA 用例记为通过。

## GPU 权限恢复后，完成剩余 20 条

1. 确认用户已恢复授权；在获准的 GPU 节点核对两张卡、可用显存、驱动、端口 10093/10094，以及是否有本项目残留 server/worker。
   只处理经核实属于本任务的进程，不使用历史 PID；如须终止 stopped 进程，先核实身份再处理，不直接 SIGCONT 旧流水线。
2. 读取 `env.sh`，确认模型、LIBERO 资源、两个 conda 环境仍完整。现有 NVIDIA 用户态 EGL 库固定为 550.90.07；
   新节点驱动不匹配时先修正匹配的图形库加载方式。`setup_graphics.py` 当前也固定该版本，不能直接用于其他版本，不改系统内核驱动。
3. 先完成上面的花屏最小复现、定位与验证，再用新标签完整运行两模型的 `libero_10`；原 60 条的视频重新采集需求同时处理。
   默认可尝试 serial cross-render，先观察真实 episode 延迟、任务进度和显存。
   单纯延长 timeout 不能作为性能问题已解决的证据；若仍卡住，保留日志诊断，不改变相机、动作、初始状态或模型精度来伪造可比结果。

```bash
cd /mnt/afs/wangyunxiangt/workspace/starVLA
source examples/attention_analysis/env.sh
# 以下命令仅在用户恢复 GPU 授权后运行。标签已存在则换新标签。
bash examples/attention_analysis/launch_all_tasks.sh \
  alltasks_seed7_libero10_resume1 libero_10 --serial-cross-render

tmux -S /tmp/starvla-attn-$(id -u).sock attach -t starvla-attn
# Ctrl-b d 分离；SSH 断开后 tmux 内继续。
tail -f "$ATTENTION_ROOT/runs/tmux/alltasks_seed7_libero10_resume1.log"
```

launcher 默认使用 GPU 0/1；若授权的是其他编号，使用 `run_rollouts --gpus <编号1> <编号2>`，仍放进 tmux。
不要重新运行原四套 launcher 标签，也不要覆盖既有 trace；已有目录会拒绝覆盖。

4. 新 suite 完成后完整验收，再组装最终 80 条：

```bash
source examples/attention_analysis/env.sh
"$ATTENTION_PYTHON" -m examples.attention_analysis.suite_report \
  "$ATTENTION_ROOT/runs/reports/alltasks_seed7_libero10_resume1"

"$ATTENTION_PYTHON" -m examples.attention_analysis.assemble_report \
  --sources "$ATTENTION_ROOT/runs/reports/alltasks_seed7_retry1" \
            "$ATTENTION_ROOT/runs/reports/alltasks_seed7_libero10_resume1" \
  --output "$ATTENTION_ROOT/runs/reports/alltasks_seed7_completed_resume1"

# 验收通过后发布；同一来源的已有 60 条任务会安全复用，新发布最后 20 条。
"$ATTENTION_PYTHON" -m examples.attention_analysis.publish_results \
  "$ATTENTION_ROOT/runs/reports/alltasks_seed7_completed_resume1"
```

后一个来源替换完整的 model/suite，不按 episode 成败筛选。先等 rollout 完成再执行上述组装，无需常驻 `--wait`。
最后必须有 80 条 success/failure 结果、0 程序错误、0 缺失；数值、图片、逐动作对齐和视频完整验收通过。
正常任务失败不影响数据完整性通过。更新交付索引、实际容量及本指南，再交付每 suite `/10` 和每模型 `/40` 的结果。

## 容量与清理边界

- 整理前已选 60 条原始数值、输入图、叠图、视频和日志合计约 0.586 GB；整理后的正式交付大小见 `runs/deliverables_reorganization.json`，原始 runs 另存。
- 原计划全部 80 条新增产物预计 0.8–2.5 GB，建议预留 5 GB；剩余两套均跑满 520 步时约产生每模型 650 个 trace，实际压缩率以完成后统计为准。
- 已有 checkpoint 与基础模型约 41.2 GB，不重新下载，不将下载缓存计为实验产物。
- 本次只清理确认重复的 pip 安装缓存/解包、conda 下载归档、图形安装包/探测分片及过期 bytecode；
  精确清单和已删除文件的逻辑字节数见 [cleanup_manifest.json](playground/attention_analysis/runs/reports/libero_current_results/cleanup_manifest.json)；实际释放空间受文件系统计量影响。
- 保留模型、归一化统计、revision、环境、LIBERO 源码/资源、实际实验 NPZ/图片/视频、失败日志、图形运行库和加载配置。
  安装归档清理后重建环境可能需要重新下载；现有环境运行不依赖这些归档。
- 用户已取消 RoboTwin，后续仅处理 LIBERO，不重新加入该 benchmark 的实验。
