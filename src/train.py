import hydra
import rootutils
import polars as pl
from pathlib import Path
from transformers import AutoConfig, AutoModel, HfApi, PreTrainedModel
from typing import Any, Dict, List, Optional, Tuple
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating litmodule <{cfg.model._target_}>")
    net: PreTrainedModel = hydra.utils.instantiate(cfg.model.net)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, processor=net.get_processor())
    datamodule.prepare_data()
    datamodule.setup("fit")
    log.info(f"Number of training samples: {datamodule.num_train_samples}")
    log.info(f"Number of validation samples: {datamodule.num_val_samples}")
    log.info(f"Number of testing samples: {datamodule.num_test_samples}")
    log.info(f"Random sample from training set -\n{datamodule.get_random_sample('train')}")
    log.info(f"Random sample from validation set -\n{datamodule.get_random_sample('validation')}")
    log.info(f"Random sample from testing set -\n{datamodule.get_random_sample('test')}")

    log.info(f"Instantiating loss function <{cfg.model.loss_fct._target_}>")
    loss_fct = hydra.utils.instantiate(
        cfg.model.loss_fct,
        class_weight=datamodule.class_weight,
        class_count=datamodule.class_count
    )

    log.info(f"Instantiating litmodule <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, net=net, loss_fct=loss_fct)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "datamodule": datamodule,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    metric_dict = {}

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=cfg.get("ckpt_path"),
        )
        metric_dict.update(trainer.callback_metrics)
        trainer.test(dataloaders=datamodule.val_dataloader(), ckpt_path="best", verbose=False)
        val_metrics = [
            {
                "Metric": k.split("/")[1],
                "Validation": v.item(),
            }
            for k, v in trainer.callback_metrics.items()
        ]
        log.info(f"Best model validation metrics:\n{pl.DataFrame(val_metrics)}")
        log.info(f"Best ckpt path: {trainer.checkpoint_callback.best_model_path}")

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(datamodule=datamodule, ckpt_path=ckpt_path, verbose=False)
        metric_dict.update(trainer.callback_metrics)
        test_metrics = [
            {
                "Metric": k.split("/")[1],
                "Test": v.item(),
            }
            for k, v in trainer.callback_metrics.items()
        ]
        log.info(f"Best model test metrics:\n{pl.DataFrame(test_metrics)}")

    if cfg.get("hf_config"):

        log.info("Loading best model for HuggingFace saving...")
        best_model = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        log.info("Registering model config and model classes to HuggingFace Auto classes...")
        model_config_class = hydra.utils.get_class(cfg.model.net.config._target_)
        model_class = hydra.utils.get_class(cfg.model.net._target_)
        AutoConfig.register(model_config_class.model_type, model_config_class)
        AutoModel.register(model_config_class, model_class)
        model_config_class.register_for_auto_class()
        model_class.register_for_auto_class("AutoModel")

        mode = cfg.hf_config.get("mode", "save")
        if mode not in ["save", "push", "save_and_push"]:
            log.warning(f"Unknown hf_config.mode: {mode}. Skipping HuggingFace saving...")

        if mode == "save" or mode == "save_and_push":
            output_dir = Path(cfg.hf_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            best_model.net.save_pretrained(output_dir)
            log.info(f"Model saved to {cfg.hf_config.output_dir} in HuggingFace format.")

        if mode == "push" or mode == "save_and_push":
            log.info("Pushing model to HuggingFace Hub...")
            output_repo = cfg.hf_config.output_repo
            private = cfg.hf_config.get("private", True)
            overwrite = cfg.hf_config.get("overwrite", True)
            api = HfApi()
            if overwrite:
                api.delete_repo(
                    repo_id=output_repo,
                    repo_type="model",
                    missing_ok=True,
                )
            api.create_repo(
                repo_id=output_repo,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            best_model.net.push_to_hub(output_repo)
            log.info(f"Model pushed to HuggingFace Hub at {output_repo}.")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
