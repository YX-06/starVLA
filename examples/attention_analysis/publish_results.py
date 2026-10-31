"""Publish complete LIBERO suites into libero-<suite>-<model>/task directories."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from examples.attention_analysis.render import episode_key

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODELS = ("libero_pi", "libero_groot")


def suite_directory(suite, model):
    if suite not in SUITES or model not in MODELS:
        raise ValueError(f"Unknown LIBERO suite/model: {suite}/{model}")
    return f"libero-{suite.removeprefix('libero_')}-{model.removeprefix('libero_')}"


def task_directory(row):
    slug = re.sub(r"[^a-z0-9]+", "_", row["task_name"].lower()).strip("_")
    return f"task{row['task_id']:02d}_{slug}"


def resolve_reference(parent, value):
    path = Path(value)
    return path if path.is_absolute() else parent / path


def transfer(source, destination):
    """Immutable binary files can share storage; every published path is a real file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise ValueError(f"Copy verification failed: {source}") from None


def publish_task(row, output, alignment):
    directory = output / suite_directory(row["task_suite"], row["model"]) / task_directory(row)
    identity = {key: row[key] for key in ("model", "task_suite", "task_id", "episode_idx", "seed", "status")}
    identity["source_run_dir"] = row["run_dir"]
    if directory.exists():
        saved = json.loads((directory / "episode.json").read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise FileExistsError(f"Refusing to replace an existing task outcome: {directory}")
        return directory
    render_file = Path(row["trace_index_path"])
    render = json.loads(render_file.read_text())
    traces = [resolve_reference(render_file.parent, value) for value in render["trace_paths"]]
    if not traces:
        raise ValueError(f"No traces for {identity}")
    context = json.loads((traces[0] / "metadata.json").read_text())["trace_context"]
    rows = [record for record in alignment if episode_key(record) == episode_key(context)]
    if len(rows) != row["execution_steps"]:
        raise ValueError(f"Incomplete task alignment: {identity}")
    directory.mkdir(parents=True)
    for trace in traces:
        metadata = json.loads((trace / "metadata.json").read_text())
        if episode_key(metadata["trace_context"]) != episode_key(context):
            raise ValueError(f"Cross-task trace: {trace}")
        for filename in ("attention.npz", "metadata.json", *(c["image_file"] for c in metadata["cameras"])):
            transfer(trace / filename, directory / "traces" / trace.name / filename)
    for image in Path(row["overlay_dir"]).glob("*.png"):
        transfer(image, directory / "overlays" / image.name)
    transfer(Path(row["video_path"]), directory / "rollout.mp4")
    transfer(Path(row["attention_video_path"]), directory / "attention.mp4")
    (directory / "alignment.jsonl").write_text("".join(json.dumps(record) + "\n" for record in rows))
    render.update(
        trace_paths=[f"traces/{trace.name}" for trace in traces], video_path="attention.mp4", overlay_dir="overlays"
    )
    (directory / "render.json").write_text(json.dumps(render, indent=2))
    episode = {
        **identity,
        "visual_quality": row.get("visual_quality", {"status": "not_checked"}),
        **{
            key: row[key]
            for key in ("task_name", "initial_state_index", "success", "execution_steps", "elapsed_seconds")
        },
        "artifacts": {
            "raw_video": "rollout.mp4",
            "attention_video": "attention.mp4",
            "traces": "traces",
            "overlays": "overlays",
            "alignment": "alignment.jsonl",
        },
    }
    (directory / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=2))
    (directory / "INDEX.md").write_text(
        f"# {row['task_name']}\n\nResult: **{row['status']}**; seed={row['seed']}; "
        f"initial state={row['initial_state_index']}; steps={row['execution_steps']}.\n\n"
        "[Rollout](rollout.mp4) · [Attention video](attention.mp4) · [NPZ and input images](traces/) · "
        "[Overlays](overlays/) · [Alignment](alignment.jsonl) · [Metadata](episode.json) · [Scale](render.json)\n\n"
        "Real token grid: 8x8. Default display averages heads, denoising steps, cross-attention layers and "
        "action queries. Cached actions reuse their source observation. Attention is not causal attribution.\n"
    )
    return directory


def refresh_indexes(output):
    tasks, groups = [], defaultdict(list)
    for path in sorted(output.glob("libero-*/*/episode.json")):
        row = json.loads(path.read_text())
        row["task_dir"] = str(path.parent.relative_to(output))
        tasks.append(row)
        groups[row["model"], row["task_suite"]].append((path.parent, row))
    suites = []
    for (model, suite), items in groups.items():
        directory = output / suite_directory(suite, model)
        counts = Counter(row["status"] for _, row in items)
        suites.append({"model": model, "task_suite": suite, "completed": len(items), **dict(counts)})
        renders, lines = [], [f"# {suite} / {model}\n", "|Task|Result|Steps|", "|---|---|---|"]
        for task, row in items:
            record = json.loads((task / "render.json").read_text())
            record.update(
                video_path=str((task / "attention.mp4").resolve()),
                overlay_dir=str((task / "overlays").resolve()),
                trace_paths=[str((task / p).resolve()) for p in record["trace_paths"]],
            )
            renders.append(record)
            lines.append(f"|[{task.name}]({task.name}/)|{row['status']}|{row['execution_steps']}|")
        (directory / "episodes.json").write_text(json.dumps(renders, indent=2))
        (directory / "INDEX.md").write_text("\n".join(lines) + "\n")
    pending = [
        {"model": model, "task_suite": suite, "task_id": task_id, "status": "pending_gpu_authorization"}
        for model in MODELS
        for suite in SUITES
        for task_id in range(10)
        if not any(r["model"] == model and r["task_suite"] == suite and r["task_id"] == task_id for r in tasks)
    ]
    summary = {
        "seed": 7,
        "target_episodes": 80,
        "completed": len(tasks),
        "pending": len(pending),
        "all_tasks_executed": len(tasks) == 80,
        "evaluation_complete": len(tasks) == 80,
        "full_evaluation_complete": (
            len(tasks) == 80 and not any(row.get("visual_quality", {}).get("status") == "failed" for row in tasks)
        ),
        "outcomes": dict(Counter(row["status"] for row in tasks)),
        "suites": suites,
        "tasks": tasks,
        "remaining_tasks": pending,
        "visual_quality_failed_tasks": sum(row.get("visual_quality", {}).get("status") == "failed" for row in tasks),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output / "episodes.csv").open("w", newline="") as stream:
        fields = ["model", "task_suite", "task_id", "task_name", "status", "seed", "execution_steps", "task_dir"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tasks + pending)
    lines = [
        "# LIBERO attention results\n",
        f"Completed: **{len(tasks)}/80**; pending: **{len(pending)}**. Single seed=7, initial_states[0].\n",
        "[All results](episodes.csv) · [Detailed summary](summary.json) · "
        "[Codex continuation guide](../../../guideline-for-codex.md)\n",
        "|Suite / model|Completed|Success|Failure|",
        "|---|---|---|---|",
    ]
    if summary["visual_quality_failed_tasks"]:
        lines.insert(
            2,
            f"**Visual quality FAILED for {summary['visual_quality_failed_tasks']} rollout videos: "
            "periodic corrupted frames at 8k+1. Execution counts below remain historical task outcomes. "
            "Numeric/decoding checks do not establish video quality. Originals are preserved.** "
            "[CPU diagnostic](../runs/diagnostics/video_flicker/REPORT.md)\n",
        )
    for row in suites:
        name = suite_directory(row["task_suite"], row["model"])
        lines.append(f"|[{name}]({name}/)|{row['completed']}/10|{row.get('success', 0)}|{row.get('failure', 0)}|")
    lines += [
        "",
        "Current execution scope and reproduction commands are recorded in the continuation guide.",
        "Normal task failures are preserved. No smoke, synthetic, or failed-retry artifacts are included.",
        "Task folders contain actual numeric/image/video files; per-task artifact references are relative.",
    ]
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")
    return summary


def publish_results(source, output):
    source, output = Path(source), Path(output)
    report = json.loads((source / "summary.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if not report["accepted"] or (manifest["seed"], manifest["episodes"], manifest["max_tasks"]) != (7, 1, -1):
        raise ValueError("Publish only validated complete suites using seed=7, one episode/task and all ten tasks")
    groups = defaultdict(list)
    for row in report["episodes"]:
        groups[row["model"], row["task_suite"]].append(row)
    for key, rows in groups.items():
        if len(rows) != 10 or {r["task_id"] for r in rows} != set(range(10)):
            raise ValueError(f"Incomplete or duplicate suite: {key}")
        if any(r["status"] not in ("success", "failure") or r["episode_idx"] != 0 for r in rows):
            raise ValueError(f"Incomplete episode: {key}")
    for rows in groups.values():
        alignment = [
            json.loads(line) for line in (Path(rows[0]["run_dir"]) / "alignment.jsonl").read_text().splitlines()
        ]
        for row in rows:
            publish_task(row, output, alignment)
    return refresh_indexes(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Validated suite_report directory")
    parser.add_argument("--output", type=Path, default=Path(os.environ["ATTENTION_ROOT"]) / "deliverables")
    args = parser.parse_args()
    result = publish_results(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("completed", "pending", "outcomes")}, indent=2))
