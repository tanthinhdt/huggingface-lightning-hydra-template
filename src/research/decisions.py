"""Threshold policies are fitted on held-out development predictions only."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def metric(y, scores, name, thresholds=0.5):
    if name == "average_precision":
        if np.any(y.sum(axis=0) == 0):
            raise ValueError("Average precision requires positive examples for every label")
        return float(average_precision_score(y, scores, average="macro"))
    if name == "auroc":
        return float(roc_auc_score(y, scores, average="macro"))
    pred = scores >= np.asarray(thresholds)
    if name in ("f1_macro", "f1_micro"):
        return float(f1_score(y, pred, average=name.split("_")[1], zero_division=0))
    if name == "mcc" and y.shape[1] == 1:
        return float(matthews_corrcoef(y.ravel(), pred.ravel()))
    raise ValueError(f"Unsupported metric: {name}")


def fit_policy(y, scores, labels, config):
    mode, objective = config["mode"], config["metric"]
    thresholds = np.full(len(labels), float(config["fallback"]))
    if mode not in ("fixed", "global", "per_label"):
        raise ValueError("Threshold mode must be fixed, global or per_label")
    if mode != "fixed" and objective not in ("f1_macro", "f1_micro", "mcc"):
        raise ValueError("Threshold fitting requires a threshold-dependent objective")
    if mode == "per_label" and objective != "f1_macro":
        raise ValueError("Independent label optimization supports macro F1 only")
    grid = np.linspace(0, 1, int(config["grid_size"]))
    if len(grid) < 2:
        raise ValueError("grid_size must be at least two")
    if mode == "global":
        values = [metric(y, scores, objective, t) for t in grid]
        thresholds[:] = grid[int(np.argmax(values))]
    elif mode == "per_label":
        for j in range(len(labels)):
            positives = y[:, j].sum()
            if min(positives, len(y) - positives) >= config["min_support"]:
                values = [
                    f1_score(y[:, j], scores[:, j] >= t, average="binary", zero_division=0)
                    for t in grid
                ]
                thresholds[j] = grid[int(np.argmax(values))]
    return {
        "labels": list(labels),
        "thresholds": thresholds.tolist(),
        "config": dict(config),
        "fitting_score": metric(y, scores, objective, thresholds),
        "score_role": "tuning_only",
    }


def evaluate_scores(y, scores, policy):
    result = {
        name: metric(y, scores, name, policy["thresholds"]) for name in ("f1_macro", "f1_micro")
    }
    for name in ("average_precision", "auroc"):
        try:
            value = metric(y, scores, name)
            result[name] = value if np.isfinite(value) else None
        except ValueError:
            result[name] = None
    if y.shape[1] == 1:
        result["mcc"] = metric(y, scores, "mcc", policy["thresholds"])
    return result
