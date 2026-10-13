"""Method plugins explain pre-threshold logits; no blanket inference_mode."""

from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from src.research.artifacts import read_json, write_json, provenance


class GradientSaliency:
    def explain(self, model, inputs, target):
        x = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            output = model(x)
            if output.ndim != 2 or not 0 <= target < output.shape[1]:
                raise ValueError("Saliency expects model logits[N,L] and a valid target label")
            (gradient,) = torch.autograd.grad(output[:, target].sum(), x)
        return gradient.detach().cpu().numpy()


class IntegratedGradients:
    def __init__(self, steps=32):
        if steps < 2:
            raise ValueError("Integrated gradients requires at least two integration points")
        self.steps = steps

    def explain(self, model, inputs, target):
        # Zero baseline in raw feature coordinates; use a custom method for other baselines.
        gradient = GradientSaliency()
        total = np.zeros_like(inputs)
        for i, alpha in enumerate(np.linspace(0, 1, self.steps)):
            weight = 0.5 if i in (0, self.steps - 1) else 1.0
            total += weight * gradient.explain(model, inputs * alpha, target)
        return inputs * total / (self.steps - 1)


class AttentionAnalysis:
    """Adapters expose get_attentions(inputs) -> tuple of [N, heads, tokens, tokens]."""

    def explain(self, model, inputs, target):
        if not hasattr(model, "get_attentions"):
            raise ValueError(
                "Attention analysis requires an adapter exposing get_attentions(inputs)"
            )
        with torch.no_grad():
            layers = model.get_attentions(inputs)
            if not layers:
                raise ValueError("Adapter returned no attention tensors")
            stacked = torch.stack(tuple(layers))
            if stacked.ndim != 5 or stacked.shape[1] != len(inputs):
                raise ValueError("Expected attention layers shaped [N, heads, tokens, tokens]")
            return stacked.mean(dim=(0, 2)).cpu().numpy()


def interpret(cfg, data, splits, directory):
    manifest = read_json(cfg.manifest)
    if (
        manifest["dataset_fingerprint"] != data.fingerprint
        or manifest["labels"] != data.labels
        or manifest["split_signature"] != splits["signature"]
    ):
        raise ValueError("Interpretation data differs from predictor provenance")
    split = cfg.interpretation.partition
    if split not in ("development", "calibration", "test"):
        raise ValueError("Choose development, calibration, or test")
    indices = splits[split][: cfg.interpretation.max_samples]
    if not indices:
        raise ValueError("Interpretation requires at least one sample")
    backend = hydra.utils.instantiate(OmegaConf.create(manifest["backend"]))
    method = hydra.utils.instantiate(cfg.interpretation.method)
    member_values = []
    for checkpoint in manifest["checkpoints"]:
        model = backend.load(checkpoint)
        member_values.append(method.explain(model, data.x[indices], cfg.interpretation.target))
    # This is a weighted mean of member-logit attributions, not ensemble-probability attribution.
    values = np.average(member_values, axis=0, weights=manifest["weights"])
    directory = Path(directory)
    import uuid

    analysis_id = uuid.uuid4().hex
    directory = directory / analysis_id
    directory.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(
        {"sample_id": [data.ids[i] for i in indices], "attribution": values.tolist()}
    ).write_parquet(directory / "attributions.parquet")
    write_json(
        directory / "manifest.json",
        {
            "predictor": str(Path(cfg.manifest).resolve()),
            "method": OmegaConf.to_container(cfg.interpretation, resolve=True),
            "sample_ids": [data.ids[i] for i in indices],
            "target_type": "attention" if isinstance(method, AttentionAnalysis) else "logit",
            "ensemble_semantics": "weighted_mean_member_attributions",
            "seed": cfg.seed,
            "environment": provenance(),
        },
    )
    return str(directory)
