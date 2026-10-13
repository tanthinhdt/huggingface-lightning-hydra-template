"""Research orchestration with explicit development, calibration and test partitions."""

from pathlib import Path

import hydra
import numpy as np
import optuna
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.research.artifacts import (
    fingerprint,
    read_json,
    write_json,
    provenance,
    save_predictions,
    source_fingerprint,
)
from src.research.data import make_folds
from src.research.decisions import metric, fit_policy, evaluate_scores


def load_data(cfg):
    return hydra.utils.call(cfg.data).validate(cfg.task)


def partitions(cfg, data, output_dir):
    path = Path(output_dir) / "splits.json"
    supplied = read_json(cfg.split.manifest_path) if cfg.split.get("manifest_path") else None
    signature = fingerprint(
        {
            "data": data.fingerprint,
            "split": OmegaConf.to_container(cfg.split),
            "seed": cfg.seed,
            "supplied": supplied,
        }
    )
    if path.exists():
        manifest = read_json(path)
        if manifest["signature"] != signature:
            raise ValueError("Split manifest differs from requested data/configuration")
        return manifest
    if supplied is not None:
        lookup = {sample_id: i for i, sample_id in enumerate(data.ids)}

        def resolve(ids):
            if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
                raise ValueError("Split IDs must be unique and present in the dataset")
            return [lookup[i] for i in ids]

        manifest = {
            "signature": signature,
            "dataset_fingerprint": data.fingerprint,
            "sample_ids": data.ids,
            "labels": data.labels,
            **{key: resolve(supplied[key]) for key in ("development", "calibration", "test")},
            "folds": [
                {key: resolve(fold[key]) for key in ("train", "validation")}
                for fold in supplied["folds"]
            ],
        }
        validate_splits(manifest, data, cfg.split.strategy == "group")
        write_json(path, manifest)
        return manifest
    indices = np.arange(len(data.ids))

    def split(indices, fraction):
        if not 0 < fraction < 1:
            raise ValueError("Test/calibration fractions must lie strictly between 0 and 1")
        if cfg.split.strategy == "group":
            if data.groups is None:
                raise ValueError("Group splitting requires data.groups")
            a, b = next(
                GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=cfg.seed).split(
                    indices, groups=data.groups[indices]
                )
            )
            return indices[a], indices[b]
        stratify = data.y[indices, 0] if cfg.task == "binary" else None
        return train_test_split(
            indices, test_size=fraction, random_state=cfg.seed, stratify=stratify
        )

    development, test = split(indices, cfg.split.test_fraction)
    # Calibration remains outside CV and every final fit, matching ensemble inference.
    if cfg.split.calibration_fraction:
        development, calibration = split(development, cfg.split.calibration_fraction)
    else:
        if cfg.decision.source != "oof":
            raise ValueError("Zero calibration_fraction requires decision.source=oof")
        calibration = np.array([], dtype=int)
    folds = make_folds(data, development, cfg.split.n_folds, cfg.seed, cfg.split.strategy)
    manifest = {
        "signature": signature,
        "dataset_fingerprint": data.fingerprint,
        "sample_ids": data.ids,
        "labels": data.labels,
        "development": development.tolist(),
        "calibration": calibration.tolist(),
        "test": test.tolist(),
        "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds],
    }
    write_json(path, manifest)
    return manifest


def validate_splits(manifest, data, check_groups=False):
    if check_groups and data.groups is None:
        raise ValueError("Group splitting requires data.groups")
    parts = [set(manifest[key]) for key in ("development", "calibration", "test")]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("Development/calibration/test partitions overlap")
    if set.union(*parts) != set(range(len(data.ids))) or not parts[0] or not parts[2]:
        raise ValueError("Partitions must cover all samples with nonempty development and test")
    held_out = []
    for fold in manifest["folds"]:
        train, val = set(fold["train"]), set(fold["validation"])
        if not train or not val or train & val or train | val != parts[0]:
            raise ValueError("Each fold must partition development data without overlap")
        held_out.extend(val)
        if check_groups and set(data.groups[list(train)]) & set(data.groups[list(val)]):
            raise ValueError("Groups overlap across fold partitions")
    if len(held_out) != len(parts[0]) or set(held_out) != parts[0]:
        raise ValueError("Every development sample must appear in validation exactly once")
    if check_groups:
        groups = [set(data.groups[list(part)]) for part in parts]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Groups overlap across development/calibration/test")


def cross_validate(cfg, data, splits, directory, trial=None):
    backend = hydra.utils.instantiate(cfg.backend)
    scores, epochs, checkpoints = [], [], []
    oof_ids, oof_y, oof_scores, fold_numbers = [], [], [], []
    for number, fold in enumerate(splits["folds"]):
        result = backend.fit(
            cfg,
            data,
            fold["train"],
            fold["validation"],
            cfg.seed + number,
            Path(directory) / f"fold-{number}",
        )
        model = backend.load(result["checkpoint"])
        validation = np.asarray(fold["validation"])
        probabilities = backend.predict(model, data, validation)
        scores.append(metric(data.y[validation], probabilities, cfg.search.metric))
        epochs.append(result["selected_epochs"])
        checkpoints.append(result["checkpoint"])
        oof_ids.extend(data.ids[i] for i in validation)
        oof_y.extend(data.y[validation].tolist())
        oof_scores.extend(probabilities.tolist())
        fold_numbers.extend([number] * len(validation))
        del model
        if trial is not None:
            trial.report(float(np.mean(scores)), step=number)
            if trial.should_prune():
                write_json(
                    Path(directory) / "status.json", {"status": "pruned", "fold_scores": scores}
                )
                raise optuna.TrialPruned()
    save_predictions(Path(directory) / "oof.parquet", oof_ids, oof_y, oof_scores, data.labels)
    write_json(Path(directory) / "oof_folds.json", dict(zip(oof_ids, fold_numbers)))
    result = {
        "objective": float(np.mean(scores)),
        "fold_scores": scores,
        "selected_epochs": epochs,
        "checkpoints": checkpoints,
        "oof_predictions": str((Path(directory) / "oof.parquet").resolve()),
        "aggregation": "mean_fold_metric",
    }
    write_json(Path(directory) / "cv.json", result)
    return result


def search(cfg, data, splits, directory):
    directory = Path(directory)
    storage = cfg.search.storage or f"sqlite:///{(directory / 'study.db').resolve()}"
    study = optuna.create_study(
        storage=storage,
        study_name=cfg.search.name,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.search.pruning_startup_trials, n_warmup_steps=1
        ),
    )
    settings = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in ("data", "model", "split", "optimization", "fit", "backend")
    }
    search_settings = OmegaConf.to_container(cfg.search, resolve=True)
    search_settings.pop("n_trials")
    signature = fingerprint(
        {
            "config": settings,
            "task": cfg.task,
            "seed": cfg.seed,
            "search": search_settings,
            "splits": splits,
            "source": source_fingerprint(),
        }
    )
    previous = study.user_attrs.get("protocol_signature")
    if previous is not None and previous != signature:
        raise ValueError(
            "Existing study belongs to a different protocol; use a new output directory/study"
        )
    study.set_user_attr("protocol_signature", signature)

    def objective(trial):
        candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for key, spec in cfg.search.parameters.items():
            if spec.type == "float":
                value = trial.suggest_float(key, spec.low, spec.high, log=spec.get("log", False))
            elif spec.type == "int":
                value = trial.suggest_int(key, spec.low, spec.high)
            elif spec.type == "categorical":
                value = trial.suggest_categorical(key, list(spec.choices))
            else:
                raise ValueError(f"Unsupported search parameter: {spec.type}")
            OmegaConf.update(candidate, key, value)
        trial_dir = directory / f"trial-{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial.set_user_attr("artifact_dir", str(trial_dir.resolve()))
        OmegaConf.save(candidate, trial_dir / "config.yaml", resolve=True)
        result = cross_validate(candidate, data, splits, trial_dir, trial)
        trial.set_user_attr("fold_scores", result["fold_scores"])
        return result["objective"]

    # n_trials is a total budget, including failed/pruned/interrupted trials.
    remaining = max(0, cfg.search.n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    best_dir = Path(study.best_trial.user_attrs["artifact_dir"])
    best_cfg = OmegaConf.load(best_dir / "config.yaml")
    result = read_json(best_dir / "cv.json")
    write_json(
        directory / "selection.json",
        {
            "trial": study.best_trial.number,
            "value": study.best_value,
            "parameters": study.best_params,
            "config_path": str((best_dir / "config.yaml").resolve()),
            "cv": result,
        },
    )
    return best_cfg, result


def nested_cross_validate(cfg, data, directory):
    """Evaluate the whole selection procedure; each outer fold owns a separate study."""
    directory = Path(directory)
    outer_folds = make_folds(
        data, np.arange(len(data.ids)), cfg.split.n_folds, cfg.seed, cfg.split.strategy
    )
    results = []
    for number, (development, test) in enumerate(outer_folds):
        outer_dir = directory / f"outer-{number}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        candidate.output_dir = str(outer_dir)
        if not cfg.split.calibration_fraction:
            if cfg.decision.source != "oof":
                raise ValueError("Without calibration use decision.source=oof")
            calibration = np.array([], dtype=int)
        elif cfg.split.strategy == "group":
            a, b = next(
                GroupShuffleSplit(
                    n_splits=1, test_size=cfg.split.calibration_fraction, random_state=cfg.seed
                ).split(development, groups=data.groups[development])
            )
            development, calibration = development[a], development[b]
        else:
            development, calibration = train_test_split(
                development,
                test_size=cfg.split.calibration_fraction,
                random_state=cfg.seed,
                stratify=data.y[development, 0] if cfg.task == "binary" else None,
            )
        inner = make_folds(data, development, cfg.split.n_folds, cfg.seed, cfg.split.strategy)
        splits = {
            "dataset_fingerprint": data.fingerprint,
            "sample_ids": data.ids,
            "labels": data.labels,
            "development": development.tolist(),
            "calibration": calibration.tolist(),
            "test": test.tolist(),
            "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in inner],
        }
        splits["signature"] = fingerprint(splits)
        write_json(outer_dir / "splits.json", splits)
        # A user-specified external database can be shared, but studies must remain distinct.
        candidate.search.name = f"{cfg.search.name}-outer-{number}"
        selected, cv = search(candidate, data, splits, outer_dir)
        manifest = refit(selected, data, splits, cv, outer_dir / "final")
        results.append(evaluate(manifest, data, splits, outer_dir / "evaluation"))
    summary = {
        "role": "outer_cv_selection_procedure",
        "outer_metrics": results,
        "means": {
            k: float(np.mean([r[k] for r in results]))
            for k in results[0]
            if all(r[k] is not None for r in results)
        },
    }
    write_json(directory / "nested_cv.json", summary)
    return summary


def refit(cfg, data, splits, cv, directory):
    directory = Path(directory)
    if cfg.decision.source == "oof" and cfg.predictor.kind != "single":
        raise ValueError(
            "OOF threshold transfer supports single refits only; ensembles require calibration"
        )
    directory.mkdir(parents=True, exist_ok=True)
    selection_signature = fingerprint(
        {
            "cv": cv,
            "predictor": OmegaConf.to_container(cfg.predictor),
            "decision": OmegaConf.to_container(cfg.decision),
            "source": source_fingerprint(),
        }
    )
    existing = directory / "predictor.json"
    if existing.exists():
        frozen = read_json(existing)
        if frozen["selection_signature"] != selection_signature:
            raise ValueError(
                "Frozen predictor already exists with different settings; choose a new final_name"
            )
        if not all(Path(path).exists() for path in frozen["checkpoints"]):
            raise ValueError("A frozen predictor checkpoint is missing; restore its artifacts")
        return existing
    backend = hydra.utils.instantiate(cfg.backend)
    fit_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    fit_cfg.fit.max_epochs = max(1, int(np.median(cv["selected_epochs"])))
    if cfg.predictor.kind == "folds":
        checkpoints = cv["checkpoints"]
    elif cfg.predictor.kind in ("single", "seeds"):
        seeds = list(cfg.predictor.seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or (cfg.predictor.kind == "single" and len(seeds) != 1)
        ):
            raise ValueError("single requires one seed; seeds requires unique seeds")
        checkpoints = [
            backend.fit(
                fit_cfg, data, splits["development"], [], seed, directory / f"seed-{seed}"
            )["checkpoint"]
            for seed in seeds
        ]
    else:
        raise ValueError("Predictor kind must be single, seeds or folds")
    weights = cfg.predictor.weights
    weights = np.ones(len(checkpoints)) if weights is None else np.asarray(weights, dtype=float)
    if (
        len(weights) != len(checkpoints)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Ensemble weights must be finite, nonnegative and match members")
    manifest = {
        "backend": OmegaConf.to_container(cfg.backend),
        "checkpoints": checkpoints,
        "weights": (weights / weights.sum()).tolist(),
        "labels": data.labels,
        "dataset_fingerprint": data.fingerprint,
        "split_signature": splits["signature"],
        "refit_epochs": fit_cfg.fit.max_epochs,
        "kind": cfg.predictor.kind,
    }
    manifest["selection_signature"] = selection_signature
    if cfg.decision.source == "calibration":
        calibration = np.asarray(splits["calibration"], dtype=int)
        if not len(calibration):
            raise ValueError(
                "Calibration decision fitting requires a nonempty calibration partition"
            )
        scores = predict(manifest, data, calibration)
        targets = data.y[calibration]
        fitting_ids = [data.ids[i] for i in calibration]
        source = "held_out_calibration"
    elif cfg.decision.source == "oof":
        if cfg.predictor.kind != "single":
            raise ValueError(
                "OOF threshold transfer supports single refits only; ensembles require calibration"
            )
        from src.research.artifacts import load_predictions

        fitting_ids, targets, scores, labels = load_predictions(cv["oof_predictions"])
        if labels != data.labels:
            raise ValueError("OOF label schema mismatch")
        source = "oof_transfer_to_refit"
    else:
        raise ValueError("decision.source must be calibration or oof")
    policy = fit_policy(targets, scores, data.labels, OmegaConf.to_container(cfg.decision))
    policy["sample_ids"] = fitting_ids
    policy["source"] = source
    manifest["decision"] = policy
    save_predictions(directory / "decision_fit.parquet", fitting_ids, targets, scores, data.labels)
    path = directory / "predictor.json"
    write_json(path, manifest)
    return path


def predict(manifest, data, indices):
    if manifest["labels"] != data.labels:
        raise ValueError("Predictor and data label schemas differ")
    if not manifest["checkpoints"] or len(manifest["checkpoints"]) != len(manifest["weights"]):
        raise ValueError("Predictor member and weight counts differ")
    backend = hydra.utils.instantiate(OmegaConf.create(manifest["backend"]))
    total = None
    for checkpoint, weight in zip(manifest["checkpoints"], manifest["weights"]):
        model = backend.load(checkpoint)
        scores = backend.predict(model, data, indices)
        if scores.shape != (len(indices), len(data.labels)) or not np.isfinite(scores).all():
            raise ValueError("Backend must return finite probabilities shaped [samples, labels]")
        total = scores * weight if total is None else total + scores * weight
    return total


def evaluate(manifest_path, data, splits, directory):
    manifest = read_json(manifest_path)
    if (
        manifest["dataset_fingerprint"] != data.fingerprint
        or manifest["split_signature"] != splits["signature"]
    ):
        raise ValueError("Predictor data/split provenance differs from this evaluation")
    indices = np.asarray(splits["test"])
    metrics_path = Path(directory) / "metrics.json"
    if metrics_path.exists() and read_json(metrics_path)["predictor"] != str(
        Path(manifest_path).resolve()
    ):
        raise ValueError(
            "Evaluation directory belongs to another predictor; choose a new evaluation_name"
        )
    scores = predict(manifest, data, indices)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_predictions(
        directory / "test.parquet",
        [data.ids[i] for i in indices],
        data.y[indices],
        scores,
        data.labels,
    )
    result = evaluate_scores(data.y[indices], scores, manifest["decision"])
    write_json(
        directory / "metrics.json",
        {
            "metrics": result,
            "role": "held_out_test",
            "predictor": str(Path(manifest_path).resolve()),
            "split_signature": splits["signature"],
            "dataset_fingerprint": data.fingerprint,
            "labels": data.labels,
        },
    )
    return result


def run(cfg):
    directory = Path(cfg.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data = load_data(cfg)
    if cfg.action == "nested_cv":
        if cfg.split.get("manifest_path"):
            raise ValueError(
                "nested_cv constructs its own folds; supplied split manifests apply to flat CV"
            )
        if not (directory / "config.yaml").exists():
            OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
            write_json(directory / "environment.json", provenance())
        return nested_cross_validate(cfg, data, directory)
    splits = partitions(cfg, data, directory)
    if not (directory / "config.yaml").exists():
        OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
        write_json(directory / "environment.json", provenance())
    if cfg.action in ("search", "run"):
        selected_cfg, cv = search(cfg, data, splits, directory)
    elif cfg.action == "cv":
        return cross_validate(cfg, data, splits, directory / "cv")
    elif cfg.action == "refit":
        selected = read_json(directory / "selection.json")
        selected_cfg = OmegaConf.load(selected["config_path"])
        cv = selected["cv"]
        selected_cfg.predictor = cfg.predictor
        selected_cfg.decision = cfg.decision
    elif cfg.action == "evaluate":
        return evaluate(cfg.manifest, data, splits, directory / cfg.evaluation_name)
    elif cfg.action == "interpret":
        from src.research.interpretation import interpret

        return interpret(cfg, data, splits, directory / "interpretation")
    else:
        raise ValueError(f"Unknown action: {cfg.action}")
    if cfg.action == "search":
        return cv
    selected_cfg.predictor = cfg.predictor
    selected_cfg.decision = cfg.decision
    manifest = refit(selected_cfg, data, splits, cv, directory / cfg.final_name)
    if cfg.final_test:
        return evaluate(manifest, data, splits, directory / cfg.evaluation_name)
    return str(manifest)
