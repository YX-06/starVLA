"""Aggregate an isolated run manifest, validate all episodes, and write JSON/CSV/index."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from examples.attention_analysis.render import episode_key
from examples.attention_analysis.report import validate_exports

# Actual dimensions from the pinned PI/GR00T checkpoints and their smoke exports.
FULL_LAYERS = {"libero_pi": list(range(36)), "libero_groot": list(range(0, 16, 2))}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_video(path, expected_frames=None):
    count = 0
    with imageio.get_reader(path) as reader:
        for frame in reader:
            if frame.ndim != 3 or not np.isfinite(frame).all():
                raise ValueError(f"Invalid frame in {path}")
            count += 1
    if count == 0 or (expected_frames is not None and count != expected_frames):
        raise ValueError(f"{path}: expected {expected_frames} frames, got {count}")
    return count


def validate_job(job, rows):
    run, delivery = Path(job["run_dir"]), Path(job["delivery_dir"])
    validation = validate_exports(run)
    aligned = read_jsonl(run / "alignment.jsonl")
    by_episode, trace_ids = defaultdict(list), set()
    for path in (run / "traces").glob("*/metadata.json"):
        m = json.loads(path.read_text())
        context = m["trace_context"]
        assert context["task_suite"] == job["task_suite"], path
        assert context["run"] == job["run_name"], path
        assert context["episode_idx"] == context["episode"], path
        if "render_gpu" in job:
            assert context.get("render_gpu") == job["render_gpu"], path
        assert m["layers"] == FULL_LAYERS[job["name"]], path
        assert m["denoise_steps"] == list(range(4)) and m["action_queries"] == list(range(8)), path
        assert m["torch_seed"] == m["cuda_seed"] == job["seed"], path
        assert [c["name"] for c in m["cameras"]] == ["primary", "wrist"], path
        assert m["trace_id"] not in trace_ids, path
        trace_ids.add(m["trace_id"])
        by_episode[episode_key(context)].append(m)
        for camera in m["cameras"]:
            with Image.open(path.parent / camera["image_file"]) as image:
                image.verify()
    alignment_by_episode = defaultdict(list)
    for row in aligned:
        alignment_by_episode[episode_key(row)].append(row)
    renders = json.loads((delivery / "episodes.json").read_text())
    rendered = {episode_key(row): row for row in renders}
    assert len(rendered) == len(renders), "Duplicate rendered episodes"
    expected_keys = set()
    videos = []
    for row in rows:
        if row["status"] not in ("success", "failure"):
            continue
        key = (job["run_name"], job["task_suite"], row["task_id"], row["episode_idx"])
        expected_keys.add(key)
        execution = alignment_by_episode[key]
        assert [r["env_step"] for r in execution] == list(range(row["execution_steps"])), row
        sources = sorted(m["trace_context"]["env_step"] for m in by_episode[key])
        assert all(m["trace_context"]["task"] == row["task_name"] for m in by_episode[key]), row
        assert sources == list(range(0, row["execution_steps"], 8)), row
        assert all(r["action_query"] == r["env_step"] % 8 for r in execution), row
        render = rendered[key]
        assert render["frames"] == row["execution_steps"] and render["traces"] == len(sources), row
        video = Path(render["video_path"])
        overlay_dir = Path(render.get("overlay_dir", video.parent))
        assert len(list(overlay_dir.glob("*_camera_*.png"))) == 2 * len(sources), video
        videos.append({"path": str(video), "frames": validate_video(video, row["execution_steps"])})
        videos.append({"path": row["video_path"], "frames": validate_video(row["video_path"], row["execution_steps"])})
        row["attention_video_path"] = str(video)
        row["trace_dir"] = str(run / "traces")
        row["overlay_dir"] = str(overlay_dir)
        row["trace_index_path"] = str(video.parent / "render.json")
    assert expected_keys <= set(by_episode), "Missing episode traces"
    if all(row["status"] in ("success", "failure") for row in rows):
        assert expected_keys == set(by_episode) == set(rendered) == set(alignment_by_episode), "Extra episodes"
    validation["videos"] = videos
    validation["full_layers_steps_queries"] = "passed"
    validation["every_inference_and_execution_step"] = "passed"
    return validation


def aggregate_job(job, manifest):
    run = Path(job["run_dir"])
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else job
    n_tasks = 10 if manifest["max_tasks"] == -1 else min(manifest["max_tasks"], 10)
    recorded = read_jsonl(run / "episodes.jsonl")
    by_key = {}
    for row in recorded:
        key = (row["task_id"], row["episode_idx"])
        if key in by_key:
            raise ValueError(f"Duplicate episode in {run}: {key}")
        if row["task_suite"] != job["task_suite"] or row["seed"] != manifest["seed"]:
            raise ValueError(f"Wrong suite/seed in {run}: {row}")
        if row["initial_state_index"] != row["episode_idx"]:
            raise ValueError(f"Unexpected initial state in {run}: {row}")
        if row["status"] not in ("success", "failure", "error"):
            raise ValueError(f"Unexpected episode status: {row}")
        if row["status"] in ("success", "failure") and row["success"] != (row["status"] == "success"):
            raise ValueError(f"Inconsistent success status: {row}")
        by_key[key] = row
    rows = []
    for task_id in range(n_tasks):
        for episode_idx in range(manifest["episodes"]):
            row = dict(
                by_key.pop(
                    (task_id, episode_idx),
                    {
                        "task_suite": job["task_suite"],
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "seed": manifest["seed"],
                        "initial_state_index": episode_idx,
                        "success": None,
                        "status": "missing",
                        "error": status.get("error", "No episode result recorded"),
                    },
                )
            )
            row.update(
                model=job["name"], run_dir=str(run), visual_quality=job.get("visual_quality", {"status": "not_checked"})
            )
            rows.append(row)
    if by_key:
        raise ValueError(f"Unexpected extra episodes in {run}: {list(by_key)}")
    counts = Counter(row["status"] for row in rows)
    validation = {"status": "not_run"}
    if recorded:
        try:
            validation = validate_job(job, rows)
        except Exception as error:
            validation = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    summary = {
        "model": job["name"],
        "task_suite": job["task_suite"],
        "expected": len(rows),
        "success": counts["success"],
        "failure": counts["failure"],
        "error": counts["error"],
        "missing": counts["missing"],
        "process_status": status["status"],
        "process_errors": int(status["status"] == "error"),
        "program_error": status.get("error"),
        "validation": validation,
        "visual_quality": job.get("visual_quality", {"status": "not_checked"}),
        "run_dir": str(run),
        "delivery_dir": job["delivery_dir"],
    }
    return rows, summary


def disk_usage(paths):
    categories, total = Counter(), 0
    for directory in paths:
        for path in Path(directory).rglob("*"):
            if not path.is_file():
                continue
            size = path.stat().st_size
            category = (
                "attention_npz"
                if path.suffix == ".npz"
                else (
                    "videos" if path.suffix == ".mp4" else "images" if path.suffix == ".png" else "metadata_logs_reports"
                )
            )
            categories[category] += size
            total += size
    return {"bytes": total, "decimal_GB": total / 1e9, "categories_bytes": dict(categories)}


def build_report(report_dir):
    report_dir = Path(report_dir)
    manifest = json.loads((report_dir / "manifest.json").read_text())
    rows, suites = [], []
    for job in manifest["jobs"]:
        try:
            episode_rows, summary = aggregate_job(job, manifest)
        except Exception as error:
            # Preserve a report even when one suite's result file is corrupt.
            reason = f"{type(error).__name__}: {error}"
            n_tasks = 10 if manifest["max_tasks"] == -1 else min(manifest["max_tasks"], 10)
            episode_rows = [
                {
                    "model": job["name"],
                    "task_suite": job["task_suite"],
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "seed": manifest["seed"],
                    "initial_state_index": episode_idx,
                    "status": "missing",
                    "success": None,
                    "error": reason,
                    "run_dir": job["run_dir"],
                }
                for task_id in range(n_tasks)
                for episode_idx in range(manifest["episodes"])
            ]
            summary = {
                "model": job["name"],
                "task_suite": job["task_suite"],
                "expected": len(episode_rows),
                "success": 0,
                "failure": 0,
                "error": 0,
                "missing": len(episode_rows),
                "process_status": "error",
                "process_errors": 1,
                "program_error": reason,
                "validation": {"status": "error", "error": reason},
            }
        rows.extend(episode_rows)
        suites.append(summary)
    models = {}
    for name in sorted({job["name"] for job in manifest["jobs"]}):
        items = [s for s in suites if s["model"] == name]
        models[name] = {
            key: sum(s.get(key, 0) for s in items)
            for key in ("expected", "success", "failure", "error", "missing", "process_errors")
        }
    structural_accepted = all(
        s.get("process_status") == "executed"
        and s.get("error") == 0
        and s.get("missing") == 0
        and s["validation"]["status"] == "passed"
        for s in suites
    )
    accepted = structural_accepted and all(s.get("visual_quality", {}).get("status") != "failed" for s in suites)
    report = {
        "accepted": accepted,
        "structural_validation_passed": structural_accepted,
        "validation_scope": "Numerical/file/alignment checks and video decoding do not establish visual correctness.",
        "scope": "Single seed; one initial state per task when episodes=1",
        "seed": manifest["seed"],
        "models": models,
        "suites": suites,
        "episodes": rows,
    }
    notes = report_dir / "run_notes.json"
    if notes.exists():
        report["run_notes"] = json.loads(notes.read_text())
    fields = [
        "model",
        "task_suite",
        "task_id",
        "task_name",
        "episode_idx",
        "seed",
        "initial_state_index",
        "status",
        "success",
        "execution_steps",
        "elapsed_seconds",
        "video_path",
        "attention_video_path",
        "trace_dir",
        "overlay_dir",
        "trace_index_path",
        "error",
        "run_dir",
    ]
    with (report_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (report_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "task_suite",
                "success",
                "expected",
                "failure",
                "error",
                "missing",
                "process_status",
                "process_errors",
                "program_error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(suites)
    paths = [p for job in manifest["jobs"] for p in (job["run_dir"], job["delivery_dir"])]
    report["disk_usage"] = disk_usage(paths)
    report["disk_usage"][
        "note"
    ] = "New run and render artifacts; excludes checkpoints, caches and this small index directory"
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# LIBERO all-task evaluation",
        "",
        f"Seed: {manifest['seed']}; accepted: {accepted}.",
        "",
        "| Model | Suite | Success | Failure | Episode errors | Process status | Missing | Structural | Visual |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for suite in suites:
        lines.append(
            f"|{suite['model']}|{suite['task_suite']}|{suite.get('success', '?')}/{suite.get('expected', '?')}|"
            f"{suite.get('failure', '?')}|{suite.get('error', '?')}|{suite.get('process_status', '?')}|"
            f"{suite.get('missing', '?')}|"
            f"{suite['validation']['status']}|{suite.get('visual_quality', {}).get('status', 'not_checked')}|"
        )
    lines += [
        "",
        *[
            f"- {name}: {m['success']}/{m['expected']} successes; {m['error']} errors; {m['missing']} missing."
            for name, m in models.items()
        ],
        "",
        f"New artifacts: {report['disk_usage']['decimal_GB']:.4f} decimal GB.",
        "",
        "[Detailed JSON](summary.json), [episode CSV](episodes.csv), [suite CSV](summary.csv), "
        "[commands and manifest](manifest.json).",
        "",
        "Single-seed evaluation with the default task order and initial_states[episode_idx]. "
        "Failure means the task did not succeed; program errors and missing results are separate.",
        "",
    ]
    if notes.exists():
        lines += ["[Run notes and previous-attempt diagnostics](run_notes.json).", ""]
    for job in manifest["jobs"]:
        run, delivery = Path(job["run_dir"]), Path(job["delivery_dir"])
        lines += [
            f"- {job['run_name']}: [run/logs]({run}), [NPZ]({run / 'traces'}), "
            f"[raw videos]({run / 'rollout_videos'}), [overlays and attention videos]({delivery})."
        ]
    lines += [
        "",
        "|Model|Suite / task / episode|Result|Raw video|Attention video|NPZ paths / scale|",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        links = [
            f"[open]({row[key]})" if row.get(key) else "—"
            for key in ("video_path", "attention_video_path", "trace_index_path")
        ]
        lines.append(
            f"|{row['model']}|{row['task_suite']} / {row['task_id']} / {row['episode_idx']}|"
            f"{row['status']}|{'|'.join(links)}|"
        )
    lines += [
        "",
        "```bash",
        "source examples/attention_analysis/env.sh",
        '"$ATTENTION_PYTHON" -m examples.attention_analysis.suite_report ' + str(report_dir),
        "```",
        "",
    ]
    (report_dir / "INDEX.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    result = build_report(args.report_dir)
    print(
        json.dumps(
            {"accepted": result["accepted"], "models": result["models"], "disk_usage": result["disk_usage"]}, indent=2
        )
    )
    raise SystemExit(0 if result["accepted"] else 1)
