"""Freeze one SHA256-selected variant per original task and perturbation class."""

import argparse
import ast
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from examples.attention_analysis_plus.prepare import BENCHMARK_REVISION

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def read_map(path):
    return ast.literal_eval(ast.parse(Path(path).read_text()).body[0].value)


def selection_hash(seed, suite, original, item):
    value = f"{seed}|{suite}|{original}|{item['category']}|{item['id']}|{item['name']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_tasks(classification, original_map, runtime_map, seed=7):
    selected = []
    for suite in SUITES:
        names = runtime_map[suite]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate runtime task names: {suite}")
        indices = {name: index for index, name in enumerate(names)}
        groups = defaultdict(list)
        official_ids = set()
        for item in classification[suite]:
            original = [base for base in original_map[suite] if item["name"].startswith(base + "_")]
            if len(original) != 1 or item["category"] not in CATEGORIES:
                raise ValueError(f"Ambiguous or unknown classification: {item}")
            if item["id"] in official_ids or item["name"] not in indices:
                raise ValueError(f"Invalid official ID/name: {item}")
            official_ids.add(item["id"])
            groups[original[0], item["category"]].append(item)
        for base in original_map[suite]:
            for category in CATEGORIES:
                candidates = groups[base, category]
                if not candidates:
                    raise ValueError(f"Missing stratum: {suite}/{base}/{category}")
                item = min(candidates, key=lambda entry: selection_hash(seed, suite, base, entry))
                selected.append(
                    {
                        "task_suite": suite,
                        "original_task": base,
                        "category": category,
                        "official_id": item["id"],
                        "task_id": indices[item["name"]],
                        "variant_name": item["name"],
                        "difficulty_level": item["difficulty_level"],
                        "selection_sha256": selection_hash(seed, suite, base, item),
                        "candidate_count": len(candidates),
                    }
                )
    selected.sort(key=lambda row: (SUITES.index(row["task_suite"]), row["task_id"]))
    return selected


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    rows = manifest["tasks"]
    if manifest["benchmark_revision"] != BENCHMARK_REVISION or manifest["seed"] != 7:
        raise ValueError("Unexpected benchmark revision or seed")
    if len(rows) != 280 or Counter(row["task_suite"] for row in rows) != Counter(dict.fromkeys(SUITES, 70)):
        raise ValueError("Manifest must contain exactly 70 tasks per suite")
    if Counter(row["category"] for row in rows) != Counter(dict.fromkeys(CATEGORIES, 40)):
        raise ValueError("Manifest must contain exactly 40 tasks per perturbation class")
    groups = {(r["task_suite"], r["original_task"], r["category"]) for r in rows}
    identities = {(r["task_suite"], r["task_id"]) for r in rows}
    if len(groups) != 280 or len(identities) != 280:
        raise ValueError("Duplicate task or stratum in manifest")
    if any(
        r["selection_sha256"]
        != selection_hash(
            7,
            r["task_suite"],
            r["original_task"],
            {
                "id": r["official_id"],
                "category": r["category"],
                "name": r["variant_name"],
            },
        )
        for r in rows
    ):
        raise ValueError("Selection digest mismatch")
    return manifest


def validate_mapping(rows, suite):
    for row in rows:
        task = suite.get_task(row["task_id"])
        if task.name != row["variant_name"] or task.bddl_file != row["variant_name"] + ".bddl":
            raise ValueError(f"Manifest/runtime task mismatch: {row} vs {task}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ["PLUS_ROOT"])
    parser.add_argument("--output", type=Path, default=root / "task_manifest.json")
    args = parser.parse_args()
    folder = root / "third_party/LIBERO-plus/libero/libero/benchmark"
    classification_file = folder / "task_classification.json"
    original_file = (
        root.parent / "attention_analysis/third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    )
    runtime_file = folder / "libero_suite_task_map.py"
    selected = select_tasks(json.loads(classification_file.read_text()), read_map(original_file), read_map(runtime_file))
    manifest = {
        "schema_version": 1,
        "scope": "280-task stratified subset; not the full LIBERO-Plus benchmark",
        "benchmark_revision": BENCHMARK_REVISION,
        "seed": 7,
        "initial_state_index": 0,
        "episodes_per_task": 1,
        "task_order_index": 0,
        "selection": "min SHA256(seed|suite|original_task|category|official_id|variant_name)",
        "execution_order": "suite spatial, object, goal, 10; ascending runtime task index within suite",
        "sources_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (classification_file, original_file, runtime_file)
        },
        "tasks": selected,
    }
    with args.output.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    load_manifest(args.output)
    print(f"Frozen {len(selected)} tasks: {args.output}")


if __name__ == "__main__":
    main()
