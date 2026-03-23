import torch
from typing import Dict
from torchmetrics import Metric
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_specificity,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_matthews_corrcoef,
)


class Metrics(Metric):
    """Base class for all metric modules."""

    def __init__(self, num_labels: int) -> None:
        """
        Initialize the Metrics module.
        
        Parameters
        ----------
        num_labels: int
            The number of labels in the classification task.
        """
        super().__init__()
        self.num_labels = num_labels
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, kingdoms: torch.Tensor) -> None:
        """
        Update metric state with a batch of predictions and labels.

        Parameters
        ----------
        predictions : torch.Tensor
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
        """
        preds = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        num_labels = self.num_labels
        average = "macro"

        acc = multiclass_accuracy(preds, labels, num_classes=num_labels)
        f1 = multiclass_f1_score(preds, labels, num_classes=num_labels, average=average)
        mcc = multiclass_matthews_corrcoef(preds, labels, num_classes=num_labels)
        pre = multiclass_precision(preds, labels, num_classes=num_labels, average=average)
        sens = multiclass_recall(preds, labels, num_classes=num_labels, average=average)
        spec = multiclass_specificity(preds, labels, num_classes=num_labels, average=average)
        bacc = 0.5 * (spec + sens)

        return {
            "acc": acc,
            "bacc": bacc,
            "f1": f1,
            "mcc": mcc,
            "pre": pre,
            "sens": sens,
            "spec": spec,
        }
