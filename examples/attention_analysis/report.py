"""Build a factual delivery index from recorded statuses; optionally wait for tmux."""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from examples.attention_analysis.background import export_environments


def validate_exports(run):
    metadata_by_id = {}
    shapes = set()
    maximum_mass = 0.0
    for path in (run / "traces").glob("*/metadata.json"):
        metadata = json.loads(path.read_text())
        metadata_by_id[metadata["trace_id"]] = metadata
        expected = tuple(len(metadata[key]) for key in ("denoise_steps", "layers", "action_queries"))
        with np.load(path.parent / "attention.npz", allow_pickle=False) as arrays:
            mass = None
            for camera in metadata["cameras"]:
                weights = arrays[camera["array_key"]]
                assert weights.shape == expected + tuple(camera["grid_hw"]), path
                assert np.isfinite(weights).all() and (weights >= 0).all(), path
                camera_mass = weights.sum(axis=(-1, -2))
                mass = camera_mass if mass is None else mass + camera_mass
                shapes.add(weights.shape)
            assert mass is not None and mass.max() <= 1.0001, path
            maximum_mass = max(maximum_mass, float(mass.max()))
    alignment = run / "alignment.jsonl"
    rows = [json.loads(line) for line in alignment.read_text().splitlines()] if alignment.exists() else []
    for row in rows:
        metadata = metadata_by_id[row["trace_id"]]
        context = metadata["trace_context"]
        assert row["episode"] == context["episode"], row
        for key in ("run", "task_suite", "task_id", "episode_idx"):
            if key in row or key in context:
                assert row.get(key) == context.get(key), (key, row)
        assert row["source_env_step"] == context["env_step"], row
        assert row["action_query"] in metadata["action_queries"], row
        assert row["env_step"] - row["source_env_step"] == row["action_query"], row
    assert metadata_by_id and rows, run
    return {
        "status": "passed",
        "traces": len(metadata_by_id),
        "execution_steps": len(rows),
        "camera_shapes": sorted(shapes),
        "maximum_combined_camera_attention_mass": maximum_mass,
        "finite_nonnegative_probabilities": True,
        "chunk_source_and_episode_alignment": "passed",
    }


def rollout_summary(run, delivery):
    status_path = run / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_executed_yet"}
    log = run / "rollout.log"
    if log.exists():
        text = log.read_text(errors="replace")
        rates = re.findall(r"Total success rate:\s*([0-9.]+)", text)
        episodes = re.findall(r"Total (?:# )?episodes:\s*(\d+)", text)
        status["task_success_rate"] = float(rates[-1]) if rates else None
        status["episodes"] = int(episodes[-1]) if episodes else None
    traces = sorted((run / "traces").glob("*/metadata.json"))
    status["attention_traces"] = len(traces)
    if status.get("client_exit_code") == 0 and status.get("status", "").startswith("executed"):
        status["export_validation"] = validate_exports(run)
    status["rollout_videos"] = [str(path) for path in run.rglob("*.mp4")]
    if traces:
        trace = traces[0].parent
        metadata = json.loads(traces[0].read_text())
        sample = delivery / "sample"
        sample.mkdir(parents=True, exist_ok=True)
        for filename in ("attention.npz", "metadata.json", *(camera["image_file"] for camera in metadata["cameras"])):
            shutil.copy2(trace / filename, sample / filename)
        status["seeds"] = {key: metadata.get(key) for key in ("torch_seed", "cuda_seed")}
        status["framework"] = metadata["framework"]
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["ATTENTION_ROOT"])
    pipeline = root / "runs/tmux/pipeline.json"
    if args.wait:
        while not pipeline.exists() or json.loads(pipeline.read_text())["status"] == "running":
            time.sleep(30)
        export_environments()
    report = {"pipeline": json.loads(pipeline.read_text()) if pipeline.exists() else {}, "stages": {}, "models": {}}
    for path in (root / "runs/tmux").glob("*.json"):
        if path.name != "pipeline.json":
            report["stages"][path.stem] = json.loads(path.read_text())
    for name in ("libero_pi", "libero_groot"):
        record = {}
        for label, path in (
            ("smoke", root / "runs" / f"{name}_smoke/verification.json"),
            ("rollout", root / "runs" / name / "status.json"),
            ("revision", root / "checkpoints" / name / "revision.json"),
        ):
            record[label] = json.loads(path.read_text()) if path.exists() else {"status": "not_executed_yet"}
        candidates = [root / "runs" / f"{name}{suffix}" for suffix in ("_final", "_seeded7", "")]
        run = next((path for path in candidates if (path / "status.json").exists()), candidates[0])
        record["primary_run"] = run.name
        record["primary_rollout"] = rollout_summary(run, root / "runs/rendered" / run.name)
        report["models"][name] = record
    delivery = root / "runs/reports/smoke_summary"
    delivery.mkdir(parents=True, exist_ok=True)
    (delivery / "verification_report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# LIBERO PI / GR00T Attention 导出交付索引",
        "",
        f"流水线状态:`{report['pipeline'].get('status', 'unknown')}`。",
        "",
        "- [详细验证报告](verification_report.json)",
        f"- [阶段日志]({root / 'runs/tmux/pipeline.log'})",
        f"- [环境版本与导出]({root / 'runs/reproducibility/environments'})",
        f"- [复现命令与数据格式]({Path(__file__).resolve().parent / 'README.md'})",
        "",
        "|组合|严格加载/动作及 RNG 一致|完整 rollout|任务成功率|attention 请求数|",
        "|---|---|---|---|---|",
    ]
    for name, record in report["models"].items():
        smoke = record["smoke"]
        smoke_status = (
            "passed"
            if smoke.get("strict_load") == "passed" and smoke.get("actions_identical") and smoke.get("rng_identical")
            else smoke.get("status", "see log")
        )
        result = record["primary_rollout"]
        lines.append(
            f"|{name}|{smoke_status}|{result.get('status')}|{result.get('task_success_rate')}|"
            f"{result['attention_traces']}|"
        )
    lines += ["", "固定 `libero_goal` 首任务: `open the middle drawer of the cabinet`; 每模型一次 episode, seed=7。", ""]
    for name, record in report["models"].items():
        run = record["primary_run"]
        rendered = root / "runs/rendered" / run
        original = root / "runs" / run
        frames = sorted(rendered.glob("*_combined.png"))
        lines += [
            f"- **{name}**: [attention 视频]({rendered / 'attention.mp4'})、"
            f"[数值与元数据样例]({rendered / 'sample'})、[完整 NPZ]({original / 'traces'})、"
            f"[原始 rollout 视频]({original / 'rollout_videos'})、[执行日志]({original / 'rollout.log'})。",
        ]
        if frames:
            lines.append(f"  [代表性叠图]({frames[0]})。")
    lines += [
        "",
        "Rollout 进程成功退出与任务成功是两个指标;任务成功/失败以各组合 rollout.log 的实际记录为准。",
        "未生成验证文件的项目没有计为通过。真实数据、PNG 和 MP4 位于对应组合目录;",
        "`synthetic_validation` 仅是明确标注的合成数值/渲染验证,不能作为真实模型 rollout。",
        "",
        "SSH 重新连接同一 GPU 主机后:",
        "",
        "```bash",
        "tmux -S /tmp/starvla-attn-10723.sock attach -t starvla-attn",
        "```",
        "",
        "数组维度为 `[denoise_step, layer, action_query, grid_h, grid_w]`; head 已平均。",
        "图像权重沿完整 key 序列归一化后截取,图像总概率质量保留原值。",
        "热力图反映图像 token attention 分配,不表示 patch 的因果贡献。",
        "",
    ]
    (delivery / "INDEX.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
