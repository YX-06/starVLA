"""Inspect chunk-boundary corruption and compare lossless encoder input with video.

The isolated-spike heuristic targets the known 8k+1 failure; it is not a general
proof of visual correctness. Review the saved boundary contact sheet as well.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def frame_mae(left, right):
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean())


def isolated_spikes(deltas):
    result = []
    for index in range(1, len(deltas) - 1):
        outer = max(deltas[index - 1], deltas[index + 2] if index + 2 < len(deltas) else 0, 1)
        if min(deltas[index], deltas[index + 1]) > max(20, 3 * outer):
            result.append(index)
    return result


def image_sequence(paths):
    previous, deltas = None, []
    for path in paths:
        with Image.open(path) as image:
            current = np.asarray(image.convert("RGB"))
        deltas.append(frame_mae(previous, current) if previous is not None else 0)
        previous = current
    return {"frames": len(deltas), "deltas": deltas, "isolated_spike_frames": isolated_spikes(deltas)}


def audit_episode(video, debug=None, output=None):
    video = Path(video)
    debug = Path(debug) if debug else None
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    events = []
    if debug:
        events = [json.loads(line) for line in (debug / "events.jsonl").read_text().splitlines()]
    encoder = {event["env_step"]: event for event in events if event["stage"] == "before_encode"}
    selected = set(range(0, 19))
    previous, deltas, differences, tiles = None, [], [], []
    with imageio.get_reader(video) as reader:
        for index, current in enumerate(reader):
            deltas.append(frame_mae(previous, current) if previous is not None else 0)
            previous = current
            if index in encoder:
                path = debug / encoder[index]["images"][0]["file"]
                with Image.open(path) as image:
                    differences.append(frame_mae(np.asarray(image), current))
            if output and index in selected:
                Image.fromarray(current).save(output / f"decoded_{index:04d}.png")
                if index < 11:
                    images = [("decoded", Image.fromarray(current))]
                    if debug:
                        for camera in ("primary", "wrist"):
                            path = debug / f"step{index:06d}_before_policy_{camera}.png"
                            if path.exists():
                                with Image.open(path) as image:
                                    images.append((camera, image.convert("RGB")))
                    tile = Image.new("RGB", (128, 148 * len(images)), "white")
                    for row, (label, image) in enumerate(images):
                        tile.paste(image.resize((128, 128)), (0, row * 148 + 20))
                        ImageDraw.Draw(tile).text((3, row * 148 + 3), f"{index}: {label}", fill="black")
                    tiles.append(tile)
    spikes = isolated_spikes(deltas)
    result = {
        "video": str(video.resolve()),
        "frames": len(deltas),
        "deltas": deltas,
        "isolated_spike_frames": spikes,
        "spike_mod8_counts": dict(Counter(index % 8 for index in spikes)),
        "encoder_frames_compared": len(differences),
        "encoder_png_vs_video_mae_max": max(differences) if differences else None,
        "encoder_png_vs_video_mae_mean": float(np.mean(differences)) if differences else None,
        "scope": "Known periodic isolated-spike diagnostic; inspect PNGs/contact sheet for visual correctness.",
    }
    if debug:
        result["buffer_audit"] = json.loads((debug / "summary.json").read_text())
        result["cameras"] = {
            camera: image_sequence(sorted(debug.glob(f"step*_before_policy_{camera}.png")))
            for camera in ("primary", "wrist")
        }
        result["context_after_requests"] = [
            {"env_step": event["env_step"], **(event.get("context") or {})}
            for event in events
            if event["stage"] == "after_policy" and event.get("inference_requested")
        ]
    if output:
        if tiles:
            sheet = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (index * 128, 0))
            sheet.save(output / "boundary_frames.png")
        (output / "audit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--debug", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_episode(args.video, args.debug, args.output)
    print(
        json.dumps(
            {key: value for key, value in audit.items() if key not in ("deltas", "cameras", "context_after_requests")},
            indent=2,
        )
    )
