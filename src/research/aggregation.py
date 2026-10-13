"""Aggregate compatible final evaluations without treating CV folds as replications."""

from pathlib import Path

import numpy as np

from src.research.artifacts import read_json, write_json


def aggregate(paths, output):
    records = [read_json(path) for path in paths]
    if not records:
        raise ValueError("Provide at least one metrics.json")
    signatures = {
        (r["dataset_fingerprint"], r["split_signature"], tuple(r["labels"])) for r in records
    }
    if len(signatures) != 1 or any(r["role"] != "held_out_test" for r in records):
        raise ValueError("Only compatible held-out evaluations can be aggregated")
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("Duplicate evaluation paths")
    result = {"n": len(records), "sources": [str(p) for p in paths], "metrics": {}}
    for key in records[0]["metrics"]:
        values = [r["metrics"][key] for r in records]
        if any(v is None for v in values):
            result["metrics"][key] = {"values": values, "mean": None, "std": None}
        else:
            result["metrics"][key] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    write_json(output, result)
    return result
