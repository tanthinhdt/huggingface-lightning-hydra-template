"""Exercise real Transformers models and Arrow datasets without accessing the Hub."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.research.artifacts import read_json, load_predictions
from src.research.huggingface import (
    HuggingFaceBackend,
    build_model,
    encode_dataset,
    ClassificationCollator,
    offline_tokenizer,
)
from src.research.huggingface_data import HuggingFaceData, from_huggingface, toy_text
from src.research.interpretation import GradientSaliency, IntegratedGradients
from src.research.workflows import run, load_data


@pytest.fixture
def hf_cfg(tmp_path, monkeypatch):
    # All tests must work with no network or cached pretrained model.
    def no_network(*args, **kwargs):
        raise AssertionError("Offline Hugging Face tests must not access the network")

    monkeypatch.setattr("requests.sessions.Session.request", no_network)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        return compose(
            config_name="research_hf",
            overrides=[
                f"output_dir={tmp_path}",
                "fit.max_epochs=1",
                "split.n_folds=2",
                "search.n_trials=1",
                "data.n_samples=40",
            ],
        )


def test_hf_search_refit_evaluate_interpret(hf_cfg):
    cfg = hf_cfg
    path = run(cfg)
    root = Path(cfg.output_dir)
    manifest = read_json(path)
    checkpoint = Path(manifest["checkpoints"][0])
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "tokenizer.json").exists()
    metadata = read_json(checkpoint / "metadata.json")
    splits = read_json(root / "splits.json")
    assert metadata["train_sample_ids"] == [splits["sample_ids"][i] for i in splits["development"]]
    assert not set(metadata["train_sample_ids"]) & set(manifest["decision"]["sample_ids"])
    for fold in splits["folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
    backend = HuggingFaceBackend()
    data = load_data(cfg)
    model = backend.load(checkpoint)
    assert model.net.config.num_labels == 1
    indices = [5, 2, 9]
    expected = backend.predict(model, data, indices)
    np.testing.assert_allclose(backend.predict(backend.load(checkpoint), data, indices), expected)
    cfg.action = "evaluate"
    assert "average_precision" in run(cfg)
    ids, y, scores, labels = load_predictions(root / "evaluation/test.parquet")
    assert ids == [data.ids[i] for i in splits["test"]]
    assert scores.shape == y.shape == (len(ids), 1)
    cfg.action = "interpret"
    for method in ("GradientSaliency", "IntegratedGradients", "AttentionAnalysis"):
        cfg.interpretation.method = {
            "_target_": f"src.research.interpretation.{method}",
            **({"steps": 4} if method == "IntegratedGradients" else {}),
        }
        cfg.interpretation.max_samples = 3
        analysis = Path(run(cfg))
        explanation = read_json(analysis / "manifest.json")
        values = pl.read_parquet(analysis / "attributions.parquet")["attribution"].to_list()
        assert np.isfinite(np.asarray(values)).all()
        assert len(values) == 3
        assert len(explanation["input_metadata"]["tokens"]) == 3
        assert np.asarray(values).shape[1] == len(explanation["input_metadata"]["tokens"][0])
    cfg.action = "run"
    assert run(cfg) == path


@pytest.mark.parametrize("kind", ["single", "seeds", "folds"])
def test_hf_multilabel_ensembles(hf_cfg, kind):
    cfg = hf_cfg
    cfg.task = "multilabel"
    cfg.data.n_labels = 2
    cfg.split.strategy = "kfold"
    cfg.predictor.kind = kind
    cfg.predictor.seeds = [0] if kind == "single" else [0, 1]
    cfg.decision.mode = "per_label"
    path = run(cfg)
    manifest = read_json(path)
    backend = HuggingFaceBackend()
    data = load_data(cfg)
    predictions = [backend.predict(backend.load(p), data, [3, 1]) for p in manifest["checkpoints"]]
    from src.research.workflows import predict

    np.testing.assert_allclose(
        predict(manifest, data, [3, 1]),
        np.average(predictions, axis=0, weights=manifest["weights"]),
    )
    cfg.action = "evaluate"
    assert "f1_micro" in run(cfg)
    cfg.action = "interpret"
    cfg.interpretation.max_samples = 2
    assert (Path(run(cfg)) / "attributions.parquet").exists()


def test_hf_partition_tokenization_and_padding():
    tokenizer = offline_tokenizer()
    data = toy_text(8)
    selected = encode_dataset(data, [5, 0], tokenizer, max_length=12)
    assert len(selected) == 2
    assert selected[0]["labels"] == data.y[5].tolist()
    assert len(selected[0]["input_ids"]) > len(selected[1]["input_ids"])
    batch = ClassificationCollator(tokenizer)([selected[0], selected[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == (2, 1)
    assert batch["attention_mask"][1, -1] == 0
    assert batch["labels"].dtype == torch.float32


def test_hf_local_json_with_hydra_data_files(hf_cfg, tmp_path):
    source = tmp_path / "examples.jsonl"
    expected = toy_text(12)
    expected.dataset.to_json(str(source))
    data = from_huggingface(
        "json",
        labels=["label-0"],
        data_files=OmegaConf.create({"train": [str(source)]}),
        cache_dir=str(tmp_path / "cache"),
    ).validate("binary")
    assert data.ids == expected.ids
    np.testing.assert_array_equal(data.y, expected.y)


def test_hf_local_dataset_and_pretrained_model(hf_cfg, tmp_path):
    source = tmp_path / "source"
    toy_text(12).dataset.save_to_disk(source)
    data = from_huggingface(str(source), labels=["label-0"], from_disk=True).validate("binary")
    assert data.ids[0] == "text-0"
    net, tokenizer = build_model(hf_cfg.model, data.labels)
    pretrained = tmp_path / "pretrained"
    net.save_pretrained(pretrained)
    tokenizer.save_pretrained(pretrained)
    hf_cfg.model.name_or_path = str(pretrained)
    hf_cfg.model.config = {}
    backend = HuggingFaceBackend()
    result = backend.fit(hf_cfg, data, list(range(8)), [8, 9], 0, tmp_path / "fit")
    adapter = backend.load(result["checkpoint"])
    assert backend.predict(adapter, data, [9, 8]).shape == (2, 1)
    assert result == backend.fit(hf_cfg, data, list(range(8)), [8, 9], 0, tmp_path / "fit")
    with pytest.raises(ValueError, match="do not match"):
        backend.fit(hf_cfg, data, list(range(7)), [8, 9], 0, tmp_path / "fit")


def test_hf_text_pairs_and_embedding_attributions(hf_cfg):
    data = toy_text(4)
    data.dataset = data.dataset.add_column("pair", ["short sample"] * 4)
    data.text_pair_column = "pair"
    backend = HuggingFaceBackend()
    result = backend.fit(hf_cfg, data, [0, 1], [2, 3], 0, Path(hf_cfg.output_dir) / "pair")
    model = backend.load(result["checkpoint"])
    inputs = backend.explanation_inputs(model, data, [2, 0])
    assert inputs["token_type_ids"].max() == 1
    saliency = GradientSaliency().explain(model, inputs, 0)
    integrated = IntegratedGradients(4).explain(model, inputs, 0)
    assert saliency.shape == integrated.shape == tuple(inputs["input_ids"].shape)
    assert np.isfinite(integrated).all()
    assert np.any(saliency > 0)
    assert np.all(saliency[inputs["attention_mask"].numpy() == 0] == 0)


def test_hf_dataset_validation():
    with pytest.raises(ValueError, match="Missing dataset columns"):
        HuggingFaceData(Dataset.from_dict({"text": ["hi"]}), ["positive"])
    dataset = Dataset.from_dict(
        {"text": ["hi", "hello"], "label": [0, 1], "sample_id": ["same", "same"]}
    )
    with pytest.raises(ValueError, match="unique"):
        HuggingFaceData(dataset, ["positive"]).validate("binary")


def test_hf_full_refit_and_nested_cv(hf_cfg):
    cfg = hf_cfg
    cfg.decision.source = "oof"
    cfg.split.calibration_fraction = 0
    path = run(cfg)
    manifest = read_json(path)
    assert manifest["decision"]["source"] == "oof_transfer_to_refit"
    metadata = read_json(Path(manifest["checkpoints"][0]) / "metadata.json")
    splits = read_json(Path(cfg.output_dir) / "splits.json")
    assert len(metadata["train_sample_ids"]) + len(splits["test"]) == cfg.data.n_samples
    cfg.action = "nested_cv"
    cfg.output_dir = str(Path(cfg.output_dir) / "nested")
    result = run(cfg)
    assert len(result["outer_metrics"]) == 2


def test_hf_head_replacement_is_explicit(hf_cfg, tmp_path):
    net, tokenizer = build_model(hf_cfg.model, ["a", "b"])
    path = tmp_path / "two_labels"
    net.save_pretrained(path)
    tokenizer.save_pretrained(path)
    hf_cfg.model.name_or_path = str(path)
    hf_cfg.model.config = {}
    with pytest.raises((RuntimeError, ValueError)):
        build_model(hf_cfg.model, ["positive"])
    hf_cfg.model.ignore_mismatched_sizes = True
    resized, _ = build_model(hf_cfg.model, ["positive"])
    assert resized.config.num_labels == 1
