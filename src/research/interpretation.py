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
        if hasattr(model, "gradient_saliency"):
            return model.gradient_saliency(inputs, target)
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
        if hasattr(model, "integrated_gradients"):
            return model.integrated_gradients(inputs, target, self.steps)
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
            batch_size = (
                inputs["input_ids"].shape[0]
                if isinstance(inputs, dict) or hasattr(inputs, "input_ids")
                else len(inputs)
            )
            if stacked.ndim != 5 or stacked.shape[1] != batch_size:
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
    input_metadata = None
    for checkpoint in manifest["checkpoints"]:
        model = backend.load(checkpoint)
        inputs = (
            backend.explanation_inputs(model, data, indices)
            if hasattr(backend, "explanation_inputs")
            else data.select_inputs(indices)
        )
        if hasattr(model, "interpretation_metadata"):
            member_metadata = model.interpretation_metadata(inputs)
            if input_metadata is not None and input_metadata != member_metadata:
                raise ValueError("Cannot average token attributions with different token mappings")
            input_metadata = member_metadata
        member_values.append(method.explain(model, inputs, cfg.interpretation.target))
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
            "input_metadata": input_metadata,
            "ensemble_semantics": "weighted_mean_member_attributions",
            "seed": cfg.seed,
            "environment": provenance(),
        },
    )
    return str(directory)
