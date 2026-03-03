import torch
from torch import nn


class Criterion(nn.Module):
    """Base class for all criterion modules."""

    def __init__(self, class_weight: torch.Tensor, class_count: torch.Tensor):
        """Initialize the criterion.
        Parameters
        ----------
        class_weight : torch.Tensor
            A tensor containing the weight for each class.
        class_count : torch.Tensor
            A tensor containing the count of samples for each class.
        """
        super().__init__()
        self.class_weight = class_weight
        self.class_count = class_count

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss given the model's output logits and the true labels.
        
        Parameters
        ----------
        logits : torch.Tensor
            The raw output from the model before applying activation functions.
        labels : torch.Tensor
            The ground truth labels for the input data.
        
        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CrossEntropyLoss(Criterion):

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss given the model's output logits and the true labels.
        
        Parameters
        ----------
        logits : torch.Tensor
            The raw output from the model before applying activation functions.
        labels : torch.Tensor
            The ground truth labels for the input data.

        Returns
        -------
        torch.Tensor
            The computed cross-entropy loss.
        """
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weight.to(logits.device))
        return loss_fct(logits, labels)
