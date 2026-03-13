import os
import torch
from typing import Any, Dict, Optional, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for the Huggingface dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size

        self.data: Optional[DatasetDict] = None
        self.class_weight: torch.Tensor = None
        self.class_count: torch.Tensor = None

    @property
    def num_train_samples(self) -> int:
        """Return the number of training samples."""
        return self.data["train"].num_rows if self.data else 0

    @property
    def num_val_samples(self) -> int:
        """Return the number of validation samples."""
        return self.data["validation"].num_rows if self.data else 0

    @property
    def num_test_samples(self) -> int:
        """Return the number of test samples."""
        return self.data["test"].num_rows if self.data else 0

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        load_dataset(self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        def process(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            """Process a batch from the dataset.

            :param batch: A batch from the dataset.
            :return: The processed batch.
            """
            inputs = self.hparams.processor.tokenize_sequences(batch["sequence"])
            labels = self.hparams.processor.encode_labels(batch["label"])
            return {**inputs, "label": labels}

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data:
            data = load_dataset(self.hparams.data_dir)
            self.data = data.map(
                process,
                batched=True,
                try_original_type=False,
            )
            self.data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Calculate class counts and weights if not calculated already
        if self.class_count is None:
            num_labels = self.hparams.processor.config.num_labels
            self.class_count = torch.bincount(torch.tensor(self.data["train"]["label"]), minlength=num_labels)
        if self.class_weight is None:
            weight = 1.0 / (self.class_count.float() + 1e-8)    # Add epsilon to avoid division by zero
            self.class_weight = weight / weight.sum()           # Normalize weights

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data["train"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data["validation"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data["test"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def get_random_sample(self, split: str = "train") -> str:
        """Get a random sample from the specified split.

        Parameters
        ----------
        split: str, optional (default="train")
            The data split to sample from. One of 'train', 'val', or 'test'.

        Returns
        -------
        str
            A string representation of the sample.
        """
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in data.")
        df = self.data[split].to_polars()
        sample = df.sample(n=1, seed=int(os.getenv("PL_GLOBAL_SEED", 42))).to_dicts()[0]
        summary = ""
        for k, v in sample.items():
            summary += f"> {k}: {v}\n"
        return summary.strip()


if __name__ == "__main__":
    _ = DataModule()
