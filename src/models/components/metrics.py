import torch
from typing import Dict
from torchmetrics import Metric


class Metrics(Metric):
    """Base class for all metric modules."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, kingdoms: torch.Tensor) -> None:
        """
        Update metric state with a batch of predictions and labels.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values. Shape: (batch_size,)
        labels : torch.Tensor
            Ground truth labels. Shape: (batch_size,)
        """
        self.predictions.append(predictions)
        self.labels.append(labels)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the metric from accumulated state.

        Returns
        -------
        Dict[str, torch.Tensor]
            Computed metric values.
            
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        preds = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        raise NotImplementedError("Please implement this method.")
