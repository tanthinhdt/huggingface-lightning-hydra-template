"""An offline example and a stable feature-data adapter contract."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

from src.research.artifacts import fingerprint


@dataclass
class ResearchData:
    x: np.ndarray
    y: np.ndarray
    ids: list
    labels: list
    groups: np.ndarray = None

    def validate(self, task):
        if self.x.shape[0] != len(self.ids) or self.y.shape != (len(self.ids), len(self.labels)):
            raise ValueError("Data must have aligned x[N,...], y[N,L], IDs[N], labels[L]")
        if len(set(self.ids)) != len(self.ids) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Sample IDs and label names must be unique")
        if not all(isinstance(value, str) for value in self.ids + self.labels):
            raise ValueError("Sample IDs and label names must be strings")
        if task not in ("binary", "multilabel") or (task == "binary" and len(self.labels) != 1):
            raise ValueError("Use binary with one output, or multilabel with L outputs")
        if not np.isin(self.y, [0, 1]).all() or not np.isfinite(self.x).all():
            raise ValueError("Expected finite inputs and binary targets")
        if self.groups is not None and len(self.groups) != len(self.ids):
            raise ValueError("Groups must align with IDs")
        return self

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "ids": self.ids,
                "labels": self.labels,
                "x": self.x.tolist(),
                "y": self.y.tolist(),
                "groups": None if self.groups is None else self.groups.tolist(),
            }
        )


def synthetic(n_samples=120, n_features=8, n_labels=1, seed=42):
    if n_labels == 1:
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=4,
            n_redundant=0,
            random_state=seed,
        )
        y = y[:, None]
    else:
        x, y = make_multilabel_classification(
            n_samples=n_samples, n_features=n_features, n_classes=n_labels, random_state=seed
        )
    return ResearchData(
        x.astype("float32"),
        y.astype("float32"),
        [f"sample-{i}" for i in range(n_samples)],
        [f"label-{i}" for i in range(n_labels)],
    )


def from_npz(path):
    """Load numeric features and targets; store IDs/labels as Unicode arrays, never objects."""
    with np.load(path, allow_pickle=False) as bundle:
        return ResearchData(
            bundle["x"].astype("float32"),
            bundle["y"].astype("float32"),
            bundle["ids"].tolist(),
            bundle["labels"].tolist(),
            bundle["groups"] if "groups" in bundle else None,
        )


def make_folds(data, indices, n_splits, seed, strategy):
    indices = np.asarray(indices)
    if strategy == "stratified":
        if data.y.shape[1] != 1:
            raise ValueError("For multilabel use kfold or supply a project-specific splitter")
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        parts = splitter.split(indices, data.y[indices, 0])
    elif strategy == "group":
        if data.groups is None or data.y.shape[1] != 1:
            raise ValueError("group requires groups and binary targets")
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        parts = splitter.split(indices, data.y[indices, 0], data.groups[indices])
    elif strategy == "kfold":
        parts = KFold(n_splits, shuffle=True, random_state=seed).split(indices)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    return [(indices[a], indices[b]) for a, b in parts]
