"""Keep Arrow datasets intact while exposing stable metadata to research protocols."""

from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import OmegaConf

from src.research.artifacts import fingerprint


@dataclass
class HuggingFaceData:
    dataset: Dataset
    labels: list
    text_column: str = "text"
    label_column: str = "label"
    id_column: str = "sample_id"
    group_column: str = None
    text_pair_column: str = None
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.dataset, Dataset):
            raise ValueError("Provide a map-style datasets.Dataset, not a DatasetDict or stream")
        required = [self.text_column, self.label_column, self.id_column]
        required += [c for c in (self.group_column, self.text_pair_column) if c]
        missing = set(required) - set(self.dataset.column_names)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        self.ids = list(self.dataset[self.id_column])
        self.y = np.asarray(self.dataset[self.label_column], dtype=np.float32)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self.groups = np.asarray(self.dataset[self.group_column]) if self.group_column else None

    def validate(self, task):
        if self.y.shape != (len(self.ids), len(self.labels)):
            raise ValueError("Targets must align with sample IDs and the label schema")
        if len(set(self.ids)) != len(self.ids) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Sample IDs and label names must be unique")
        if not all(isinstance(v, str) for v in self.ids + list(self.labels)):
            raise ValueError("Sample IDs and label names must be strings")
        if task not in ("binary", "multilabel") or (task == "binary" and len(self.labels) != 1):
            raise ValueError("Use binary with one output, or multilabel with L outputs")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Use numeric 0/1 targets or multilabel indicator vectors")
        for column in [self.text_column, self.text_pair_column]:
            if column and not all(isinstance(t, str) for t in self.dataset[column]):
                raise ValueError("Text inputs must be strings")
        return self

    def select_inputs(self, indices):
        return self.dataset.select([int(i) for i in indices])

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "arrow_fingerprint": self.dataset._fingerprint,
                "ids": self.ids,
                "labels": list(self.labels),
                "targets": self.y.tolist(),
                "source": self.source,
                "columns": [
                    self.text_column,
                    self.text_pair_column,
                    self.label_column,
                    self.id_column,
                    self.group_column,
                ],
            }
        )


def from_huggingface(
    path,
    labels,
    name=None,
    split="train",
    revision=None,
    data_files=None,
    cache_dir=None,
    from_disk=False,
    text_column="text",
    label_column="label",
    id_column="sample_id",
    group_column=None,
    text_pair_column=None,
):
    """Load one indexed HF dataset; published train/test partitions use a split manifest."""
    if from_disk:
        dataset = load_from_disk(path)
        if not isinstance(dataset, Dataset):
            dataset = dataset[split]
    else:
        if OmegaConf.is_config(data_files):
            data_files = OmegaConf.to_container(data_files, resolve=True)
        dataset = load_dataset(
            path,
            name=name,
            split=split,
            revision=revision,
            data_files=data_files,
            cache_dir=cache_dir,
        )
    return HuggingFaceData(
        dataset,
        list(labels),
        text_column,
        label_column,
        id_column,
        group_column,
        text_pair_column,
        {
            "path": str(path),
            "name": name,
            "split": split,
            "revision": revision,
            "from_disk": from_disk,
        },
    )


def toy_text(n_samples=80, n_labels=1):
    """Deterministic in-memory text data, with no Hub or network access."""
    targets = [[(i // (2**j)) % 2 for j in range(n_labels)] for i in range(n_samples)]
    texts = [
        " ".join(("good" if value else "bad") + " sample" for value in row)
        + " "
        + "short " * (i % 3)
        for i, row in enumerate(targets)
    ]
    dataset = Dataset.from_dict(
        {"sample_id": [f"text-{i}" for i in range(n_samples)], "text": texts, "label": targets}
    )
    return HuggingFaceData(
        dataset, [f"label-{j}" for j in range(n_labels)], source={"kind": "offline_toy"}
    )
