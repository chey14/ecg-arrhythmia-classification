
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    gamma controls how much extra weight is given to
    difficult examples. Optional class weights help with
    class imbalance.
    """

    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma

        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        # Cross-entropy for each example, before reduction.
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        # Probability assigned to the correct class.
        pt = torch.exp(-ce_loss)

        # Down-weight easy examples.
        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]

        return loss.mean()
