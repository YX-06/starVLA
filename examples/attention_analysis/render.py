"""Render exported attention with one probability color scale across cameras.

Run: python -m examples.attention_analysis.render TRACE_ROOT --output OUTPUT
Indices refer to saved semantic indices, not positions in the selected arrays.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
from matplotlib import colormaps


def restore_map(weights, transforms):
    """Undo resize/padding/right-angle rotation for display on an original image.

    This interpolation changes display resolution only; quantitative mass must
    always be computed on the original exported token grid.
    """
    image = Image.fromarray(weights.astype(np.float32))
    for transform in reversed(transforms):
        target = tuple(transform["resize_to_wh"])
        image = image.resize(target, Image.Resampling.BILINEAR)
        left, top, right, bottom = transform.get("padding", [0, 0, 0, 0])
        image = image.crop((left, top, image.width - right, image.height - bottom))
        image = image.resize(tuple(transform["resize_from_wh"]), Image.Resampling.BILINEAR)
        angle = transform.get("rotation_degrees", 0)
        if angle % 90:
            raise ValueError("Only right-angle inverse rotations are supported")
        image = image.rotate(-angle, expand=True)
    return np.asarray(image)


def select_mean(array, metadata, steps=None, layers=None, queries=None):
    for axis, (key, selected) in enumerate((("denoise_steps", steps), ("layers", layers), ("action_queries", queries))):
        if selected is not None:
            positions = [metadata[key].index(value) for value in selected]
            array = np.take(array, positions, axis=axis)
    return array.mean(axis=(0, 1, 2))


def render_trace(trace, output, *, steps=None, layers=None, queries=None, vmax=None):
    trace, output = Path(trace), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((trace / "metadata.json").read_text())
    with np.load(trace / "attention.npz", allow_pickle=False) as arrays:
        maps = [
            select_mean(arrays[camera["array_key"]], metadata, steps, layers, queries) for camera in metadata["cameras"]
        ]
    vmax = float(vmax if vmax is not None else max(weights.max() for weights in maps))
    tiles = []
    for index, (camera, weights) in enumerate(zip(metadata["cameras"], maps, strict=False)):
        image = Image.open(trace / camera["image_file"]).convert("RGB")
        displayed = np.asarray(Image.fromarray(weights).resize(image.size, Image.Resampling.BILINEAR))
        color = (colormaps["inferno"](displayed / max(vmax, 1e-12))[..., :3] * 255).astype(np.uint8)
        overlay = Image.blend(image, Image.fromarray(color), alpha=0.5)
        canvas = Image.new("RGB", (max(image.width, 400), image.height + 90), "white")
        canvas.paste(overlay, (0, 90))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), f"{camera['name']} | {metadata['framework']}", fill="black")
        draw.text((4, 20), f"grid={camera['grid_hw']} mass={weights.sum():.5f}", fill="black")
        draw.text((4, 37), f"display scale: 0..{vmax:.6g} (raw probability)", fill="black")
        context = metadata["trace_context"]
        draw.text((4, 54), f"source observation: step {context.get('env_step', '?')}", fill="black")
        draw.text((4, 71), "Bilinear display; attention is not causal attribution", fill="black")
        canvas.save(output / f"{metadata['trace_id']}_camera_{index}.png")
        tiles.append(canvas)
    combined = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
    offset = 0
    for tile in tiles:
        combined.paste(tile, (offset, 0))
        offset += tile.width
    combined.save(output / f"{metadata['trace_id']}_combined.png")
    return combined, metadata


def episode_key(context):
    """Identity includes task and suite because episode indices restart for every task."""
    return (
        context.get("run"),
        context.get("task_suite"),
        context.get("task_id"),
        context.get("episode_idx", context.get("episode", 0)),
    )


def render_exports(traces, output, *, alignment=None, fps=5, vmax=None, **options):
    traces, output = Path(traces), Path(output)
    paths = [traces] if (traces / "metadata.json").exists() else [p.parent for p in traces.glob("*/metadata.json")]
    paths.sort(key=lambda p: json.loads((p / "metadata.json").read_text())["request_index"])
    if not paths:
        raise ValueError("No exported traces found")
    groups, trace_keys = {}, {}
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text())
        key = episode_key(metadata["trace_context"])
        groups.setdefault(key, []).append((path, metadata))
        trace_keys[metadata["trace_id"]] = key
    rows_by_group = {}
    if alignment:
        with Path(alignment).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("trace_id") not in trace_keys:
                    raise ValueError(f"Alignment references missing trace: {row}")
                key = trace_keys[row["trace_id"]]
                if key != episode_key(row):
                    raise ValueError(f"Cross-episode alignment: {row}")
                rows_by_group.setdefault(key, []).append(row)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for key, items in groups.items():
        # Preserve the historical single-episode output path for legacy traces.
        grouped = len(groups) > 1 or key[2] is not None
        destination = output / f"task{key[2]:02d}_episode{key[3]:03d}" if grouped else output
        destination.mkdir(parents=True, exist_ok=True)
        scale = vmax
        if scale is None:
            scale = 0.0
            for path, metadata in items:
                with np.load(path / "attention.npz", allow_pickle=False) as arrays:
                    scale = max(scale, *(float(select_mean(a, metadata, **options).max()) for a in arrays.values()))
        # Write PNGs immediately; at most one combined frame is retained in memory.
        frames = {}
        for path, metadata in items:
            render_trace(path, destination, vmax=scale, **options)
            frames[metadata["trace_id"]] = destination / f"{metadata['trace_id']}_combined.png"
        frame_count, cached_id, cached_frame = 0, None, None
        with imageio.get_writer(destination / "attention.mp4", fps=fps) as writer:
            rows = rows_by_group.get(key, []) if alignment else [{"trace_id": m["trace_id"]} for _, m in items]
            if not rows:
                raise ValueError(f"No aligned execution frames for episode {key}")
            for row in rows:
                if cached_id != row["trace_id"]:
                    with Image.open(frames[row["trace_id"]]) as image:
                        cached_frame = image.convert("RGB")
                    cached_id = row["trace_id"]
                frame = cached_frame.copy()
                if alignment:
                    ImageDraw.Draw(frame).text(
                        (4, frame.height - 16),
                        f"execution step={row['env_step']} chunk query={row['action_query']} "
                        f"source step={row['source_env_step']}",
                        fill="cyan",
                    )
                writer.append_data(np.asarray(frame))
                frame_count += 1
        record = {
            "run": key[0],
            "task_suite": key[1],
            "task_id": key[2],
            "episode_idx": key[3],
            "vmin": 0,
            "vmax": float(scale),
            "frames": frame_count,
            "traces": len(items),
            "trace_paths": [str(path.resolve()) for path, _ in items],
            "video_path": str((destination / "attention.mp4").resolve()),
            **options,
        }
        (destination / "render.json").write_text(json.dumps(record, indent=2))
        records.append(record)
    (output / "episodes.json").write_text(json.dumps(records, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--queries", type=int, nargs="+")
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--alignment", type=Path, help="Client JSONL; repeat cached frames with source/query labels")
    args = parser.parse_args()
    render_exports(
        args.traces,
        args.output,
        alignment=args.alignment,
        fps=args.fps,
        vmax=args.vmax,
        **{key: getattr(args, key) for key in ("steps", "layers", "queries")},
    )


if __name__ == "__main__":
    main()
