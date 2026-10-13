"""Transformers + Datasets backend for binary/multilabel sequence classification."""

from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

from src.research.artifacts import (
    fingerprint,
    source_fingerprint,
    read_json,
    write_json,
    save_predictions,
)
from src.research.huggingface_data import HuggingFaceData


def offline_tokenizer():
    """A fixed vocabulary, never fitted on validation/test samples."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "sample", "short"]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )


def build_model(config, labels):
    if config.name_or_path:
        tokenizer_path = config.tokenizer_name_or_path or config.name_or_path
        tokenizer_revision = config.tokenizer_revision
        if tokenizer_revision is None and tokenizer_path == config.name_or_path:
            tokenizer_revision = config.revision
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, revision=tokenizer_revision, local_files_only=config.local_files_only
        )
        model_config = AutoConfig.from_pretrained(
            config.name_or_path, revision=config.revision, local_files_only=config.local_files_only
        )
        for key, value in OmegaConf.to_container(config.config, resolve=True).items():
            setattr(model_config, key, value)
    else:
        tokenizer = offline_tokenizer()
        options = OmegaConf.to_container(config.config, resolve=True)
        model_type = options.pop("model_type", "bert")
        model_config = AutoConfig.for_model(model_type, vocab_size=len(tokenizer), **options)
    if tokenizer.pad_token_id is None:
        raise ValueError("Sequence batching requires a tokenizer with a configured padding token")
    if config.max_length < 2:
        raise ValueError("model.max_length must be at least two")
    positions = getattr(model_config, "max_position_embeddings", None)
    if positions is not None and config.max_length > positions:
        raise ValueError("model.max_length exceeds the architecture's position embedding capacity")
    model_config.num_labels = len(labels)
    model_config.id2label = dict(enumerate(labels))
    model_config.label2id = {label: i for i, label in enumerate(labels)}
    model_config.problem_type = "multi_label_classification"
    model_config.pad_token_id = tokenizer.pad_token_id
    if config.name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.name_or_path,
            config=model_config,
            revision=config.revision,
            local_files_only=config.local_files_only,
            attn_implementation=config.attn_implementation,
            ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", False),
        )
    else:
        model = AutoModelForSequenceClassification.from_config(
            model_config, attn_implementation=config.attn_implementation
        )
    return model, tokenizer


class HFModule(LightningModule):
    """Lightning controls fitting; Transformers supplies logits and portable serialization."""

    def __init__(self, net, lr):
        super().__init__()
        self.net, self.lr = net, lr

    def forward(self, **inputs):
        return self.net(**inputs).logits

    def _loss(self, batch):
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        return nn.functional.binary_cross_entropy_with_logits(
            self(**inputs), batch["labels"].float()
        )

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self._loss(batch), batch_size=len(batch["labels"]), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class ClassificationCollator:
    """Pad model inputs dynamically; keep binary labels shaped [batch,1]."""

    def __init__(self, tokenizer):
        self.padding = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.padding(inputs)
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def encode_dataset(data, indices, tokenizer, max_length, include_labels=True):
    subset = data.select_inputs(indices)

    def encode(batch):
        pair = batch[data.text_pair_column] if data.text_pair_column else None
        encoded = tokenizer(
            batch[data.text_column],
            text_pair=pair,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        if include_labels:
            targets = np.asarray(batch[data.label_column], dtype=np.float32)
            encoded["labels"] = targets.reshape(len(targets), -1).tolist()
        return dict(encoded)

    return subset.map(
        encode,
        batched=True,
        remove_columns=subset.column_names,
        desc="Tokenizing selected partition",
    )


class HFModelAdapter:
    """A loaded predictor owns the exact tokenizer and interpretation capabilities."""

    def __init__(self, net, tokenizer, metadata):
        self.net, self.tokenizer, self.metadata = net.eval(), tokenizer, metadata

    def encode(self, dataset):
        pair_column = self.metadata["text_pair_column"]
        return self.tokenizer(
            list(dataset[self.metadata["text_column"]]),
            text_pair=list(dataset[pair_column]) if pair_column else None,
            padding=True,
            truncation=True,
            max_length=self.metadata["max_length"],
            return_tensors="pt",
        )

    def interpretation_metadata(self, inputs):
        token_ids = inputs["input_ids"].tolist()
        return {
            "tokens": [self.tokenizer.convert_ids_to_tokens(row) for row in token_ids],
            "input_ids": token_ids,
            "attention_mask": inputs["attention_mask"].tolist(),
            "attribution_units": "token",
            "special_tokens_included": True,
            "gradient_saliency_reduction": "l2_embedding_gradient",
            "integrated_gradients_baseline": "zero_word_embeddings",
            "integrated_gradients_reduction": "sum_embedding_dimensions",
        }

    def get_attentions(self, inputs):
        output = self.net(**inputs, output_attentions=True)
        if output.attentions is None:
            raise ValueError(
                "This architecture/attention implementation does not expose attention weights"
            )
        mask = (
            inputs["attention_mask"][:, None, :, None] * inputs["attention_mask"][:, None, None, :]
        )
        return tuple(layer * mask for layer in output.attentions)

    def _embedding_inputs(self, inputs):
        if not hasattr(self.net, "get_input_embeddings"):
            raise ValueError("This model does not expose input embeddings")
        embeddings = self.net.get_input_embeddings()(inputs["input_ids"]).detach()
        kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
        return embeddings, kwargs

    def _gradient(self, embeddings, kwargs, target):
        embeddings = embeddings.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.net(inputs_embeds=embeddings, **kwargs).logits
            if not 0 <= target < logits.shape[1]:
                raise ValueError("Interpretation target is outside the label schema")
            (gradient,) = torch.autograd.grad(logits[:, target].sum(), embeddings)
        return gradient.detach()

    def gradient_saliency(self, inputs, target):
        embeddings, kwargs = self._embedding_inputs(inputs)
        gradient = self._gradient(embeddings, kwargs, target)
        values = gradient.norm(dim=-1) * inputs["attention_mask"]
        return values.cpu().numpy()

    def integrated_gradients(self, inputs, target, steps):
        embeddings, kwargs = self._embedding_inputs(inputs)
        total = torch.zeros_like(embeddings)
        for i, alpha in enumerate(torch.linspace(0, 1, steps)):
            total += (0.5 if i in (0, steps - 1) else 1.0) * self._gradient(
                embeddings * alpha, kwargs, target
            )
        values = (embeddings * total / (steps - 1)).sum(dim=-1) * inputs["attention_mask"]
        return values.cpu().numpy()


class HuggingFaceBackend:
    """One single-device fit per fold; no model/tokenizer downloads when reloading artifacts."""

    def fit(self, cfg, data, train_ids, validation_ids, seed, output_dir):
        if not isinstance(data, HuggingFaceData):
            raise ValueError("HuggingFaceBackend requires HuggingFaceData")
        data.validate(cfg.task)
        if cfg.fit.devices != 1 or cfg.fit.get("num_nodes", 1) != 1:
            raise ValueError("Hugging Face research backend currently supports one device per fit")
        train_ids = np.asarray(train_ids, dtype=int)
        validation_ids = np.asarray(validation_ids, dtype=int)
        for indices in (train_ids, validation_ids):
            if (
                len(set(indices)) != len(indices)
                or np.any(indices < 0)
                or np.any(indices >= len(data.ids))
            ):
                raise ValueError("Fit indices must be unique and within dataset bounds")
        if not len(train_ids) or set(train_ids) & set(validation_ids):
            raise ValueError("Training must be nonempty and disjoint from validation")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
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
                "seed": int(seed),
            }
        )
        if (directory / "result.json").exists():
            previous = read_json(directory / "result.json")
            if previous["signature"] != signature:
                raise ValueError(
                    "Existing fit artifacts do not match requested configuration/data"
                )
            if (Path(previous["checkpoint"]) / "metadata.json").exists():
                return previous
        write_json(directory / "status.json", {"status": "running", "signature": signature})
        try:
            seed_everything(int(seed), workers=True)
            net, tokenizer = build_model(cfg.model, data.labels)
            module = HFModule(net, cfg.optimization.lr)

            def loader(indices, shuffle=False):
                dataset = encode_dataset(data, indices, tokenizer, cfg.model.max_length)
                return DataLoader(
                    dataset,
                    batch_size=cfg.optimization.batch_size,
                    shuffle=shuffle,
                    collate_fn=ClassificationCollator(tokenizer),
                )

            checkpoint = ModelCheckpoint(
                dirpath=directory / "checkpoints", monitor="val/loss", mode="min"
            )
            trainer = Trainer(
                **OmegaConf.to_container(cfg.fit, resolve=True),
                callbacks=[checkpoint] if len(validation_ids) else [],
                enable_checkpointing=bool(len(validation_ids)),
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=directory
            )
            trainer.fit(
                module,
                loader(train_ids, True),
                loader(validation_ids) if len(validation_ids) else None,
            )
            selected_epochs = cfg.fit.max_epochs
            if len(validation_ids):
                selected = torch.load(
                    checkpoint.best_model_path, map_location="cpu", weights_only=False
                )
                module.load_state_dict(selected["state_dict"])
                selected_epochs = int(selected["epoch"]) + 1
            portable = directory / "huggingface"
            module.net.cpu().eval().save_pretrained(portable, safe_serialization=True)
            tokenizer.save_pretrained(portable)
            metadata = {
                "labels": data.labels,
                "task": cfg.task,
                "max_length": cfg.model.max_length,
                "text_column": data.text_column,
                "text_pair_column": data.text_pair_column,
                "model_source": OmegaConf.to_container(cfg.model, resolve=True),
                "resolved_model_revision": getattr(net.config, "_commit_hash", None),
                "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                "dataset_source": data.source,
                "dataset_fingerprint": data.fingerprint,
                "train_sample_ids": [data.ids[i] for i in train_ids],
                "validation_sample_ids": [data.ids[i] for i in validation_ids],
                "prediction_batch_size": cfg.optimization.batch_size,
                "probability_transform": "sigmoid",
            }
            write_json(portable / "metadata.json", metadata)
            result = {
                "checkpoint": str(portable.resolve()),
                "selected_epochs": int(selected_epochs),
                "signature": signature,
                "seed": int(seed),
            }
            if len(validation_ids):
                scores = self.predict(self.load(portable), data, validation_ids)
                save_predictions(
                    directory / "validation.parquet",
                    [data.ids[i] for i in validation_ids],
                    data.y[validation_ids],
                    scores,
                    data.labels,
                )
            OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
            write_json(directory / "result.json", result)
            write_json(directory / "status.json", {"status": "complete"})
            return result
        except Exception as exc:
            write_json(directory / "status.json", {"status": "failed", "error": repr(exc)})
            raise

    def load(self, checkpoint):
        metadata = read_json(Path(checkpoint) / "metadata.json")
        net = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        return HFModelAdapter(net, tokenizer, metadata)

    def predict(self, model, data, indices):
        if model.metadata["labels"] != data.labels:
            raise ValueError("Predictor and dataset label schemas differ")
        if (
            model.metadata["text_column"] != data.text_column
            or model.metadata["text_pair_column"] != data.text_pair_column
        ):
            raise ValueError("Predictor and dataset input columns differ")
        dataset = encode_dataset(
            data, indices, model.tokenizer, model.metadata["max_length"], include_labels=False
        )
        loader = DataLoader(
            dataset,
            batch_size=model.metadata["prediction_batch_size"],
            collate_fn=ClassificationCollator(model.tokenizer),
        )
        scores = []
        with torch.no_grad():
            for batch in loader:
                scores.append(model.net(**batch).logits.sigmoid().cpu().numpy())
        return np.concatenate(scores) if scores else np.empty((0, len(data.labels)))

    def explanation_inputs(self, model, data, indices):
        return model.encode(data.select_inputs(indices))
