"""Validate every trace/action/video and publish a factual stratified-subset index."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from examples.attention_analysis.render import render_exports
from examples.attention_analysis.visual_audit import isolated_spikes
from examples.attention_analysis_plus.client import unnormalize_actions
from examples.attention_analysis_plus.prepare import BENCHMARK_REVISION, MODEL_REVISION, MODEL_SHA256
from examples.attention_analysis_plus.run import task_directory, write_json
from examples.attention_analysis_plus.tasks import CATEGORIES, SUITES, load_manifest


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def video_quality(path, expected):
    previous, deltas = None, []
    with imageio.get_reader(path) as reader:
        for frame in reader:
            require(frame.ndim == 3 and frame.shape[2] == 3, f"Invalid decoded frame: {path}")
            deltas.append(0.0 if previous is None else float(np.abs(frame.astype(float) - previous).mean()))
            previous = frame.astype(float)
    require(len(deltas) == expected, f"Video frame count differs: {path}: {len(deltas)} != {expected}")
    spikes = isolated_spikes(deltas)
    return {
        "frames": len(deltas),
        "decodable": True,
        "isolated_spike_frames": spikes,
        "visual_quality": "suspected_corruption" if spikes else "no_isolated_spikes_detected",
        "scope": "Heuristic only; no repair or frame filtering; not a general visual-correctness guarantee",
    }


def validate_task(task_run, suite_run, episode, stats):
    # Incremental CPU rendering and the final report may meet at the same task.
    # Keep locks on local storage: flock is unreliable on some AFS mounts.
    locks = Path("/tmp") / f"starvla-plus-validation-{os.getuid()}"
    locks.mkdir(exist_ok=True)
    name = hashlib.sha256(str(task_run.resolve()).encode()).hexdigest()
    with (locks / name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _validate_task(task_run, suite_run, episode, stats)


def _validate_task(task_run, suite_run, episode, stats):
    rows = read_rows(task_run / "alignment.jsonl")
    count = episode["execution_steps"]
    require(count > 0 and len(rows) == count, "Missing action alignment rows")
    trace_ids, requests, shapes = set(), [], set()
    traces = task_run / "traces"
    traces.mkdir(exist_ok=True)
    for step, row in enumerate(rows):
        require(row["env_step"] == step, "Non-contiguous execution steps")
        require(
            row["task_id"] == episode["task_id"] and row["task_suite"] == episode["task_suite"], "Cross-task alignment"
        )
        require(
            row["episode_idx"] == 0 and row["action_query"] == step % 8 and row["source_env_step"] == step - step % 8,
            "Invalid action query/source alignment",
        )
        trace_id = row["trace_id"]
        require(trace_id and isinstance(trace_id, str) and Path(trace_id).name == trace_id, "Invalid trace ID")
        source = suite_run / "traces" / trace_id
        metadata = json.loads((source / "metadata.json").read_text())
        require(metadata["framework"] == "Qwen_PI_v4", "Unexpected framework")
        provenance = metadata["provenance"]
        require(
            provenance["revision"] == MODEL_REVISION
            and provenance["checkpoint_sha256"] == MODEL_SHA256
            and provenance["benchmark_revision"] == BENCHMARK_REVISION,
            "Incorrect pinned model/benchmark",
        )
        require(metadata["torch_seed"] == metadata["cuda_seed"] == 7, "Incorrect policy server seed")
        context = metadata["trace_context"]
        require(
            context["selection"] == {key: episode[key] for key in context["selection"]},
            "Trace selection differs from episode",
        )
        require(
            context["task_id"] == row["task_id"]
            and context["task_suite"] == row["task_suite"]
            and context["env_step"] == row["source_env_step"]
            and context["episode_idx"] == 0
            and context["run"] == row["run"],
            "Cross-task/source trace reference",
        )
        require(
            metadata["layers"] == list(range(28))
            and metadata["denoise_steps"] == list(range(4))
            and metadata["action_queries"] == list(range(8)),
            "Missing layer/denoise/query samples",
        )
        with np.load(task_run / "chunks" / f"{trace_id}.npz", allow_pickle=False) as chunk:
            actions, normalized = chunk["actions"], chunk["normalized"]
            require(actions.shape == normalized.shape == (8, 7), "Invalid saved chunk shape")
            require(np.isfinite(actions).all() and np.isfinite(normalized).all(), "Nonfinite action chunk")
            np.testing.assert_array_equal(actions, unnormalize_actions(normalized, stats))
            np.testing.assert_array_equal(actions[step % 8], row["raw_action"])
            expected = np.concatenate(
                (
                    np.asarray(actions[step % 8, :6], dtype=np.float32),
                    [np.float32(1 - 2 * (float(actions[step % 8, 6]) > 0.5))],
                )
            )
            np.testing.assert_array_equal(expected, row["executed_action"])
        if trace_id in trace_ids:
            require(trace_id == rows[step - 1]["trace_id"] and step % 8 != 0, "Reused trace across chunks")
            continue
        require(step % 8 == 0, "Chunk starts at a nonzero query")
        trace_ids.add(trace_id)
        require([c["name"] for c in metadata["cameras"]] == ["primary", "wrist"], "Incorrect camera mapping")
        with np.load(source / "attention.npz", allow_pickle=False) as arrays:
            require(set(arrays.files) == {"camera_0", "camera_1"}, "Missing camera NPZ")
            mass = np.zeros((4, 28, 8), dtype=np.float32)
            for camera in metadata["cameras"]:
                weights = arrays[camera["array_key"]]
                require(weights.shape == (4, 28, 8, *camera["grid_hw"]), "Unexpected attention shape")
                require(np.isfinite(weights).all() and (weights >= 0).all(), "Invalid probabilities")
                mass += weights.sum((-1, -2))
                shapes.add(weights.shape)
                with Image.open(source / camera["image_file"]) as image:
                    image.load()
                    require(image.size == tuple(camera["input_size_wh"]), "Input camera resolution mismatch")
            require((mass <= 1.0001).all(), "Image probability mass exceeds full-key softmax mass")
        target = traces / trace_id
        if not target.exists():
            shutil.copytree(source, target, copy_function=os.link)
        requests.append(
            {
                "trace_id": trace_id,
                "bytes": sum(p.stat().st_size for p in source.iterdir()),
                "npz_bytes": (source / "attention.npz").stat().st_size,
            }
        )
    require(len(trace_ids) == (count + 7) // 8, "Trace count differs from executed chunk count")
    camera_quality = {
        name: video_quality(task_run / filename, count)
        for name, filename in (("primary", "rollout.mp4"), ("wrist", "rollout_wrist.mp4"))
    }
    if not (task_run / "attention.mp4").exists():
        records = render_exports(traces, task_run / "rendered", alignment=task_run / "alignment.jsonl", fps=10)
        require(len(records) == 1, "Cross-task render grouping")
        rendered = Path(records[0]["video_path"]).parent
        (rendered / "attention.mp4").replace(task_run / "attention.mp4")
        overlays = task_run / "overlays"
        overlays.mkdir()
        for path in rendered.glob("*.png"):
            path.replace(overlays / path.name)
        render = {
            **records[0],
            "video_path": "attention.mp4",
            "trace_paths": [f"traces/{trace_id}" for trace_id in sorted(trace_ids)],
        }
        write_json(task_run / "render.json", render)
    video_quality(task_run / "attention.mp4", count)
    require(
        len(list((task_run / "overlays").glob("*_camera_*.png"))) == 2 * len(trace_ids), "Missing per-camera overlays"
    )
    return {
        "passed": True,
        "trace_ids": sorted(trace_ids),
        "requests": requests,
        "camera_shapes": sorted(shapes),
        "camera_quality": camera_quality,
        "actions_and_chunks_aligned": True,
        "shared_episode_camera_scale": True,
    }


def summarize(episodes, selected):
    expected_keys = {(r["task_suite"], r["task_id"]) for r in selected}
    keys = [(r["task_suite"], r["task_id"]) for r in episodes]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected_keys):
        raise ValueError("Duplicate or unselected result")
    result = {
        "scope": "280-task stratified subset; not the full LIBERO-Plus benchmark",
        "expected": len(selected),
        "recorded": len(episodes),
        "missing": len(expected_keys - set(keys)),
        "successes": sum(r["status"] == "success" for r in episodes),
        "normal_failures": sum(r["status"] == "failure" for r in episodes),
        "program_errors": sum(r["status"] not in ("success", "failure") for r in episodes),
    }
    for name, field, values in (("by_suite", "task_suite", SUITES), ("by_category", "category", CATEGORIES)):
        result[name] = {
            value: {
                "successes": sum(r["status"] == "success" and r[field] == value for r in episodes),
                "normal_failures": sum(r["status"] == "failure" and r[field] == value for r in episodes),
                "program_errors": sum(r["status"] not in ("success", "failure") and r[field] == value for r in episodes),
                "recorded": sum(r[field] == value for r in episodes),
                "expected": sum(r[field] == value for r in selected),
            }
            for value in values
        }
    return result


def build_report(output, manifest_path, publish=False):
    started = time.monotonic()
    output, manifest_path = Path(output), Path(manifest_path)
    manifest = load_manifest(manifest_path)
    state = json.loads((output / "status.json").read_text())
    selected = manifest["tasks"][:1] if state["smoke"] else manifest["tasks"]
    episodes, validations, errors, all_ids = [], {}, {}, set()
    checkpoint = Path(os.environ["PLUS_ROOT"]) / "checkpoints/libero_plus_pi"
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())["franka"]["action"]
    for suite in SUITES:
        path = output / suite / "episodes.jsonl"
        if path.exists():
            episodes.extend(read_rows(path))
    expected_by_key = {(r["task_suite"], r["task_id"]): r for r in selected}
    for episode in episodes:
        key = (episode["task_suite"], episode["task_id"])
        name = f"{key[0]}/{task_directory(episode)}"
        try:
            require(key in expected_by_key, "Unselected episode")
            require(all(episode[k] == v for k, v in expected_by_key[key].items()), "Episode selection mismatch")
            require(episode["status"] in ("success", "failure"), episode.get("error", "Program error"))
            require(episode["success"] is (episode["status"] == "success"), "Inconsistent success flag")
            require(
                episode["seed"] == 7 and episode["initial_state_index"] == episode["episode_idx"] == 0,
                "Incorrect episode protocol",
            )
            limit = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[key[0]]
            require(0 < episode["execution_steps"] <= limit, "Incorrect action-step limit")
            if episode["status"] == "failure":
                require(episode["execution_steps"] == limit, "Failure recorded before the normal step limit")
            validation = validate_task(output / name, output / episode["task_suite"], episode, stats)
            require(not (all_ids & set(validation["trace_ids"])), "Trace reused across tasks")
            all_ids.update(validation["trace_ids"])
            validations[name] = validation
            write_json(output / name / "validation.json", validation)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    source_ids = {p.parent.name for suite in SUITES for p in (output / suite / "traces").glob("*/metadata.json")}
    if source_ids != all_ids:
        errors["trace_coverage"] = "Server traces are missing or not validated against an executed chunk"
    report = summarize(episodes, selected)
    report.update(
        smoke=state["smoke"],
        elapsed_seconds=state.get("elapsed_seconds"),
        validation_errors=errors,
        validations=validations,
        manifest=str(manifest_path),
        structural_validation_passed=not errors and report["missing"] == report["program_errors"] == 0,
    )
    report["evaluation_complete"] = report["structural_validation_passed"] and not state["smoke"]
    report["visual_quality"] = dict(
        Counter(
            camera["visual_quality"]
            for validation in validations.values()
            for camera in validation["camera_quality"].values()
        )
    )
    report["visual_repair"] = "cancelled by user; all frames retained, no repair or postprocessing"
    requests = [r for validation in validations.values() for r in validation["requests"]]
    files = [p.stat() for p in output.rglob("*") if p.is_file()]
    unique = {(stat.st_dev, stat.st_ino): stat for stat in files}
    report["storage"] = {
        "artifact_logical_bytes": sum(stat.st_size for stat in files),
        "artifact_unique_bytes": sum(stat.st_size for stat in unique.values()),
        "artifact_allocated_bytes": sum(stat.st_blocks * 512 for stat in unique.values()),
        "request_count": len(requests),
        "mean_request_bytes": float(np.mean([r["bytes"] for r in requests])) if requests else None,
        "mean_request_npz_bytes": float(np.mean([r["npz_bytes"] for r in requests])) if requests else None,
    }
    report["validation_and_render_seconds"] = time.monotonic() - started
    report["total_elapsed_seconds"] = (state.get("elapsed_seconds") or 0) + report["validation_and_render_seconds"]
    write_json(output / "summary.json", report)
    fields = [
        "task_suite",
        "official_id",
        "task_id",
        "variant_name",
        "original_task",
        "category",
        "difficulty_level",
        "status",
        "success",
        "execution_steps",
        "elapsed_seconds",
    ]
    with (output / "episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)
    lines = [
        "# LIBERO-Plus PI attention",
        "",
        report["scope"],
        "",
        f"Successes: {report['successes']}/{report['expected']}; normal failures: {report['normal_failures']}; "
        f"program errors: {report['program_errors']}; missing: {report['missing']}.",
        "",
        "[JSON](summary.json) | [CSV](episodes.csv)",
        "",
        f"Artifact bytes (unique files): {report['storage']['artifact_unique_bytes']}; "
        f"rollout/model startup seconds: {report.get('elapsed_seconds')}; "
        f"validation/render seconds: {report['validation_and_render_seconds']:.2f}.",
        "",
        f"[Reproduction commands]({Path(__file__).with_name('README.md').resolve()}) | "
        f"[Run log]({output.parent / 'tmux' / (output.name + '.log')})",
        "",
        "| Suite | Successes |",
        "| --- | --- |",
    ]
    lines += [f"| {suite} | {r['successes']}/{r['expected']} |" for suite, r in report["by_suite"].items()]
    lines += ["", "| Perturbation class | Successes |", "| --- | --- |"]
    lines += [f"| {category} | {r['successes']}/{r['expected']} |" for category, r in report["by_category"].items()]
    lines += ["", "Camera quality is reported by a heuristic; corrupt frames are preserved without repair.", ""]
    for episode in episodes:
        directory = f"libero-plus-{episode['task_suite'].removeprefix('libero_')}-pi/{task_directory(episode)}"
        lines.append(f"- [{episode['variant_name']}]({directory}/episode.json): {episode['status']}")
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")
    if publish:
        require(report["evaluation_complete"], "Cannot publish incomplete/error results")
        delivery = Path(os.environ["PLUS_ROOT"]) / "deliverables"
        delivery.mkdir(exist_ok=True)
        for episode in episodes:
            name = task_directory(episode)
            target = delivery / f"libero-plus-{episode['task_suite'].removeprefix('libero_')}-pi" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copytree(
                output / episode["task_suite"] / name,
                target,
                copy_function=os.link,
                ignore=shutil.ignore_patterns("rendered"),
            )
            record = {**episode, "video_path": "rollout.mp4", "wrist_video_path": "rollout_wrist.mp4"}
            # Replace hardlinks before editing so raw run metadata is preserved.
            write_json(target / "episode.json", record)
        for name in ("summary.json", "episodes.csv", "INDEX.md"):
            shutil.copy2(output / name, delivery / name)
        shutil.copy2(manifest_path, delivery / "task_manifest.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("validations",)}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(os.environ["PLUS_ROOT"]) / "task_manifest.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = build_report(args.run, args.manifest, args.publish)
    raise SystemExit(0 if report["structural_validation_passed"] else 1)


if __name__ == "__main__":
    main()
