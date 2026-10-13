from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.research.artifacts import read_json, load_predictions, save_predictions
from src.research.decisions import fit_policy
from src.research.interpretation import GradientSaliency, IntegratedGradients
from src.research.training import LightningBackend
from src.research.workflows import run, partitions, load_data
from src.research.aggregation import aggregate


def test_search_refit_evaluate_interpret_resume(research_cfg):
    cfg = research_cfg
    manifest_path = run(cfg)
    root = Path(cfg.output_dir)
    manifest = read_json(manifest_path)
    splits = read_json(root / "splits.json")
    parts = [set(splits[k]) for k in ("development", "calibration", "test")]
    assert all(not parts[i] & parts[j] for i in range(3) for j in range(i))
    assert set.union(*parts) == set(range(cfg.data.n_samples))
    assert not (root / "evaluation").exists()
    assert manifest["decision"]["source"] == "held_out_calibration"
    assert len(manifest["checkpoints"]) == 1
    assert len(read_json(root / "trial-0/cv.json")["fold_scores"]) == 2
    cfg.action = "evaluate"
    metrics = run(cfg)
    assert 0 <= metrics["f1_macro"] <= 1
    ids, y, scores, labels = load_predictions(root / "evaluation/test.parquet")
    assert ids == [splits["sample_ids"][i] for i in splits["test"]]
    assert y.shape == scores.shape == (len(ids), len(labels))
    cfg.action = "interpret"
    analysis_dir = run(cfg)
    assert (Path(analysis_dir) / "attributions.parquet").exists()
    cfg.action = "run"
    assert run(cfg) == manifest_path
    import optuna

    study = optuna.load_study(study_name=cfg.search.name, storage=f"sqlite:///{root / 'study.db'}")
    assert len(study.trials) == 1
    cfg.data.seed += 1
    with pytest.raises(ValueError, match="Split manifest"):
        run(cfg)


@pytest.mark.parametrize("kind", ["seeds", "folds"])
def test_ensembles_and_multilabel(research_cfg, kind):
    cfg = research_cfg
    cfg.task = "multilabel"
    cfg.data.n_labels = 3
    cfg.split.strategy = "kfold"
    cfg.predictor.kind = kind
    cfg.predictor.seeds = [0, 1]
    cfg.decision.mode = "per_label"
    manifest_path = run(cfg)
    manifest = read_json(manifest_path)
    assert len(manifest["checkpoints"]) == 2
    assert len(manifest["decision"]["thresholds"]) == 3
    cfg.action = "evaluate"
    assert "f1_micro" in run(cfg)
    cfg.action = "interpret"
    assert (
        read_json(Path(run(cfg)) / "manifest.json")["ensemble_semantics"]
        == "weighted_mean_member_attributions"
    )


def test_fold_isolation_preprocessing_and_recovery(research_cfg, tmp_path):
    data = load_data(research_cfg)
    backend = LightningBackend()
    result = backend.fit(
        research_cfg, data, list(range(40)), list(range(40, 50)), 0, tmp_path / "fit"
    )
    model = backend.load(result["checkpoint"])
    np.testing.assert_allclose(model.mean.numpy(), data.x[:40].mean(axis=0))
    a = backend.predict(model, data, [40, 41])
    b = backend.predict(backend.load(result["checkpoint"]), data, [40, 41])
    np.testing.assert_array_equal(a, b)
    assert result == backend.fit(
        research_cfg, data, list(range(40)), list(range(40, 50)), 0, tmp_path / "fit"
    )
    with pytest.raises(ValueError, match="do not match"):
        backend.fit(research_cfg, data, list(range(39)), list(range(40, 50)), 0, tmp_path / "fit")


def test_group_partitions(research_cfg):
    cfg = research_cfg
    data = load_data(cfg)
    data.groups = np.repeat(np.arange(30), 4)
    cfg.split.strategy = "group"
    splits = partitions(cfg, data, cfg.output_dir)
    groups = [set(data.groups[splits[k]]) for k in ("development", "calibration", "test")]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    for fold in splits["folds"]:
        assert not set(data.groups[fold["train"]]) & set(data.groups[fold["validation"]])


def test_thresholds_and_rare_label_fallback():
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
    scores = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    config = dict(mode="per_label", metric="f1_macro", fallback=0.5, grid_size=101, min_support=2)
    policy = fit_policy(y, scores, ["a", "b"], config)
    assert 0.2 < policy["thresholds"][0] <= 0.3
    assert policy["thresholds"][1] == 0.5
    config["metric"] = "f1_micro"
    with pytest.raises(ValueError, match="macro F1"):
        fit_policy(y, scores, ["a", "b"], config)


def test_interpretation_linear_ground_truth():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]]))
    x = np.array([[1.0, 4.0]], dtype="float32")
    np.testing.assert_allclose(GradientSaliency().explain(model, x, 0), [[2.0, -3.0]])
    np.testing.assert_allclose(IntegratedGradients(8).explain(model, x, 0), [[2.0, -12.0]])


def test_prediction_alignment_and_aggregation(research_cfg, tmp_path):
    run(research_cfg)
    research_cfg.action = "evaluate"
    run(research_cfg)
    path = tmp_path / "evaluation/metrics.json"
    assert aggregate([path], tmp_path / "summary.json")["n"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([path, path], tmp_path / "bad.json")
    with pytest.raises(ValueError, match="dimensions"):
        save_predictions(tmp_path / "bad.parquet", ["a"], [[0]], [[0.2, 0.3]], ["x"])


def test_config_composition():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        for name in ("train", "eval", "research"):
            cfg = compose(config_name=name)
            assert cfg.model and cfg.data
        for predictor in ("single", "seeds", "folds"):
            cfg = compose(
                config_name="research",
                overrides=[f"predictor={predictor}", "interpretation=integrated_gradients"],
            )
            assert cfg.predictor.kind == predictor


def test_full_development_oof_and_separate_actions(research_cfg):
    cfg = research_cfg
    cfg.action = "search"
    cfg.split.calibration_fraction = 0
    cfg.decision.source = "oof"
    run(cfg)
    cfg.action = "refit"
    manifest_path = run(cfg)
    manifest = read_json(manifest_path)
    splits = read_json(Path(cfg.output_dir) / "splits.json")
    assert len(splits["development"]) + len(splits["test"]) == cfg.data.n_samples
    assert manifest["decision"]["source"] == "oof_transfer_to_refit"
    cfg.action = "run"
    assert run(cfg) == manifest_path
    cfg.predictor.kind = "seeds"
    cfg.final_name = "ensemble"
    with pytest.raises(ValueError, match="ensembles require calibration"):
        run(cfg)


def test_nested_cv(research_cfg):
    research_cfg.action = "nested_cv"
    result = run(research_cfg)
    assert len(result["outer_metrics"]) == 2
    assert result["role"] == "outer_cv_selection_procedure"


def test_attention_capability():
    from src.research.interpretation import AttentionAnalysis

    with pytest.raises(ValueError, match="get_attentions"):
        AttentionAnalysis().explain(torch.nn.Linear(2, 1), np.ones((1, 2)), 0)

    class Adapter:
        def get_attentions(self, inputs):
            return (torch.ones(len(inputs), 2, 3, 3),)

    assert AttentionAnalysis().explain(Adapter(), np.ones((2, 3)), 0).shape == (2, 3, 3)


def test_published_split_manifest(research_cfg, tmp_path):
    from src.research.artifacts import write_json

    cfg = research_cfg
    data = load_data(cfg)
    original = partitions(cfg, data, tmp_path / "original")
    supplied = {
        key: [data.ids[i] for i in original[key]] for key in ("development", "calibration", "test")
    }
    supplied["folds"] = [
        {key: [data.ids[i] for i in fold[key]] for key in ("train", "validation")}
        for fold in original["folds"]
    ]
    path = tmp_path / "published.json"
    write_json(path, supplied)
    cfg.split.manifest_path = str(path)
    result = partitions(cfg, data, tmp_path / "published")
    assert result["test"] == original["test"]
    supplied["test"].append(supplied["development"][0])
    write_json(path, supplied)
    with pytest.raises(ValueError, match="overlap"):
        partitions(cfg, data, tmp_path / "bad")
    with pytest.raises(ValueError, match="Split manifest"):
        partitions(cfg, data, tmp_path / "published")


def test_predictor_variants_keep_evaluations(research_cfg):
    cfg = research_cfg
    first = run(cfg)
    cfg.action = "evaluate"
    run(cfg)
    cfg.action = "refit"
    cfg.predictor.kind = "seeds"
    cfg.predictor.seeds = [0, 1]
    with pytest.raises(ValueError, match="Frozen predictor"):
        run(cfg)
    cfg.final_name = "ensemble"
    second = run(cfg)
    assert second != first
    cfg.action = "evaluate"
    with pytest.raises(ValueError, match="evaluation_name"):
        run(cfg)
    cfg.evaluation_name = "ensemble_evaluation"
    assert "f1_micro" in run(cfg)
