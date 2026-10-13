"""One fresh Lightning fit; preprocessing is fitted within its training partition."""

from pathlib import Path

import hydra
import numpy as np
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.research.artifacts import (
    read_json,
    save_predictions,
    write_json,
    fingerprint,
    source_fingerprint,
)


class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )


class FeatureModule(LightningModule):
    def __init__(self, model_config, input_dim, output_dim, lr, mean, scale):
        super().__init__()
        self.net = hydra.utils.instantiate(
            OmegaConf.create(model_config), input_dim=input_dim, output_dim=output_dim
        )
        self.lr = lr
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.mean) / self.scale)

    def training_step(self, batch, batch_idx):
        return nn.functional.binary_cross_entropy_with_logits(self(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        loss = nn.functional.binary_cross_entropy_with_logits(self(batch[0]), batch[1])
        self.log("val/loss", loss, batch_size=len(batch[0]), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LightningBackend:
    """Replace this backend for sequence/image tasks; preserve its fit/load/predict contract."""

    def fit(self, cfg, data, train_ids, validation_ids, seed, output_dir):
        return fit_one(cfg, data, train_ids, validation_ids, seed, output_dir)

    def load(self, checkpoint):
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        module = FeatureModule(**bundle["init"])
        module.load_state_dict(bundle["state_dict"])
        return module.eval()

    def predict(self, model, data, indices):
        outputs = []
        with torch.no_grad():
            for chunk in np.array_split(np.asarray(indices), max(1, (len(indices) + 255) // 256)):
                if len(chunk):
                    outputs.append(model(torch.tensor(data.x[chunk])).sigmoid().cpu().numpy())
        return np.concatenate(outputs)


def fit_one(cfg, data, train_ids, validation_ids, seed, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids, validation_ids = np.asarray(train_ids), np.asarray(validation_ids)
    if not len(train_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("Training must be nonempty and disjoint from validation")
    signature = fingerprint(
        {
            "config": {
                key: OmegaConf.to_container(cfg[key], resolve=True)
                for key in ("model", "optimization", "fit", "backend")
            },
            "source": source_fingerprint(),
            "data": data.fingerprint,
            "train": train_ids.tolist(),
            "validation": validation_ids.tolist(),
            "seed": seed,
        }
    )
    result_path = output_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["signature"] != signature:
            raise ValueError("Existing fit artifacts do not match requested configuration/data")
        if Path(result["checkpoint"]).exists():
            return result
    if cfg.fit.devices != 1 or cfg.fit.get("num_nodes", 1) != 1:
        raise ValueError(
            "The feature backend supports one device per fit; provide a distributed backend for DDP"
        )
    seed_everything(int(seed), workers=True)
    init = {
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "input_dim": data.x.shape[1],
        "output_dim": data.y.shape[1],
        "lr": cfg.optimization.lr,
        "mean": data.x[train_ids].mean(axis=0).tolist(),
        "scale": np.maximum(data.x[train_ids].std(axis=0), 1e-6).tolist(),
    }
    module = FeatureModule(**init)

    def loader(indices, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(data.x[indices]), torch.tensor(data.y[indices])),
            batch_size=cfg.optimization.batch_size,
            shuffle=shuffle,
        )

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", monitor="val/loss", mode="min"
    )
    trainer = Trainer(
        **OmegaConf.to_container(cfg.fit, resolve=True),
        callbacks=[checkpoint] if len(validation_ids) else [],
        enable_checkpointing=bool(len(validation_ids)),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=output_dir
    )
    write_json(output_dir / "status.json", {"status": "running", "signature": signature})
    try:
        trainer.fit(
            module,
            loader(train_ids, True),
            loader(validation_ids) if len(validation_ids) else None,
        )
        if len(validation_ids):
            selected = torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=False
            )
            module.load_state_dict(selected["state_dict"])
            selected_epochs = int(selected["epoch"]) + 1
        else:
            selected_epochs = cfg.fit.max_epochs
        portable = output_dir / "model.pt"
        torch.save(
            {"init": init, "state_dict": {k: v.cpu() for k, v in module.state_dict().items()}},
            portable,
        )
        result = {
            "checkpoint": str(portable.resolve()),
            "selected_epochs": selected_epochs,
            "signature": signature,
            "seed": int(seed),
        }
        if len(validation_ids):
            scores = LightningBackend().predict(module.cpu().eval(), data, validation_ids)
            save_predictions(
                output_dir / "validation.parquet",
                [data.ids[i] for i in validation_ids],
                data.y[validation_ids],
                scores,
                data.labels,
            )
        OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
        write_json(result_path, result)
        write_json(output_dir / "status.json", {"status": "complete"})
        return result
    except Exception as exc:
        write_json(output_dir / "status.json", {"status": "failed", "error": repr(exc)})
        raise
