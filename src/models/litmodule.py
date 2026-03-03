import torch
from typing import Any, Dict, Tuple
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Metric


class LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metric: Metric,
        best_metric_name: str,
        compile: bool,
    ) -> None:
        """Initialize a `LitModule`.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model to be trained.
        criterion : torch.nn.Module
            The loss function to optimize during training.
        optimizer : torch.optim.Optimizer
            The optimizer to use for updating model parameters.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler to adjust the learning rate during training.
        metric : Metric
            The metric to evaluate model performance on validation and test sets.
        best_metric_name : str
            The name of the metric to track for best performance during validation.
        compile : bool
            Whether to compile the model using `torch.compile` for potential speedup.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # metric objects for calculating and averaging accuracy across batches
        self.val_metrics = self.hparams.metric(average="macro")
        self.test_metrics = self.hparams.metric(average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation metric
        self.val_metric_best = MaxMetric()

    def forward(self, *args, **kwargs) -> Any:
        """Perform a forward pass through the model.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Model inputs including:
            - input_ids: Tokenized protein sequences
            - attention_mask: Mask for padding tokens
            - labels: Ground truth labels for the task (optional, used for loss computation)
        
        Returns
        -------
        Any
            Tuple of (loss, logits, predictions) from the model.
        """
        return self.net(*args, **kwargs)

    def model_step(self, *args, **kwargs) -> Any:
        """Prepare inputs and perform a single forward pass.
        
        This method standardizes the input format and calls self.forward().

        Parameters
        ----------
        inputs : Dict[str, Any]
            Raw batch inputs containing sequences, kingdoms, etc.
        labels : Dict[str, torch.Tensor], optional
            Ground truth labels for the batch, by default None
            
        Returns
        -------
        Any
            Tuple of (loss, logits, predictions) from the model.
        """
        return self.forward(*args, **kwargs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_metric_best.reset()

    def on_train_epoch_start(self) -> None:
        """Reset training metrics at the start of each training epoch."""
        self.train_loss.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch containing "input_ids", "attention_mask", and "labels" tensors.
        batch_idx : int
            The index of the current batch.
        
        Returns
        -------
        torch.Tensor
            The computed loss (required for backpropagation).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        model_output = self.model_step(input_ids, attention_mask)
        logits = model_output.logits

        loss = self.hparams.criterion(logits, batch["label"])
        self.train_loss.update(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at the end of each epoch."""
        loss = self.train_loss.compute()
        self.log("train/loss", loss, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at the start of each validation epoch.
        
        This ensures we only accumulate predictions from the current epoch.
        Best metrics are NOT reset as they track maximum values across all epochs.
        """
        self.val_loss.reset()
        self.val_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch containing "input_ids", "attention_mask", and "labels" tensors.
        batch_idx : int
            The index of the current batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        model_output = self.model_step(input_ids, attention_mask, labels=labels)

        # update and log metrics
        self.val_loss.update(model_output.loss)
        self.val_metrics.update(model_output.predictions, labels)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()
        self.log("val/loss", loss, prog_bar=True)

        metrics = self.val_metrics.compute()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)

        self.val_metric_best.update(metrics[self.hparams.best_metric_name])
        self.log(f"val/{self.hparams.best_metric_name}_best", self.val_metric_best.compute(), prog_bar=True)

    def on_test_epoch_start(self) -> None:
        """Reset test metrics at the start of testing.
        
        This ensures we only accumulate predictions from the current test run.
        """
        self.test_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch containing "input_ids", "attention_mask", and "labels" tensors.
        batch_idx : int
            The index of the current batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        model_output = self.model_step(input_ids, attention_mask)
        self.test_metrics.update(model_output.predictions, batch["label"])

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at the end of testing."""
        metrics = self.test_metrics.compute()
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Parameters
        ----------
        stage : str
            The stage being set up. Either "fit", "validate", "test", or "predict".
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the learning rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitModule(None, None, None, None, None, None, None)
