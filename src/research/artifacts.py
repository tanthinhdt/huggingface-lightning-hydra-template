"""Portable JSON/Parquet artifacts and execution provenance."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    """Reject cached training results after implementation changes, including uncommitted code."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_fingerprint": source_fingerprint(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


def save_predictions(path, ids, targets, scores, labels, fold=-1):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.shape != targets.shape or scores.shape != (len(ids), len(labels)):
        raise ValueError("Prediction dimensions must match sample IDs and label schema")
    if (
        len(set(ids)) != len(ids)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("IDs must be unique and probabilities finite within [0,1]")
    pl.DataFrame(
        {
            "sample_id": ids,
            "fold": [fold] * len(ids),
            "target": targets.tolist(),
            "score": scores.tolist(),
        }
    ).write_parquet(path)
    write_json(str(path) + ".json", {"labels": list(labels), "score_type": "probability"})


def load_predictions(path):
    frame = pl.read_parquet(path)
    schema = read_json(str(path) + ".json")
    return (
        frame["sample_id"].to_list(),
        np.array(frame["target"].to_list()),
        np.array(frame["score"].to_list()),
        schema["labels"],
    )
