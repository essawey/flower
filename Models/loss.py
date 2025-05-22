from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig


# Categorical Cross Entropy Loss
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return F.cross_entropy(outputs, targets.argmax(dim=1))


class CombinedLoss(nn.Module):
    def __init__(self, config: DictConfig, mode: str, from_logits: bool, smooth: int, ignore_index: int, log_loss: bool):
        super().__init__()
        self.config = config
        self.mode = mode
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.log_loss = log_loss

        self.cross_entropy_loss = CategoricalCrossEntropyLoss()
        self.DiceLoss = config["DiceLoss"]
        self.FocalLoss = config["FocalLoss"]
        self.LovaszLoss = config["LovaszLoss"]

    def get_losses(self, outputs, targets):
        return {
            "CrossEntropyLoss": self.cross_entropy_loss(outputs, targets).item(),
            "DiceLoss": self.DiceLoss(outputs, targets).item(),
            "FocalLoss": self.FocalLoss(outputs, targets).item(),
            "LovaszLoss": self.LovaszLoss(outputs, targets).item(),
        }
    
    def forward(self, outputs, targets):
        return (
            self.cross_entropy_loss(outputs, targets) +
            self.DiceLoss(outputs, targets) + 
            self.FocalLoss(outputs, targets) +  # FIXME: You can cilp the value of the normalized focal loss [-1,1] to match the DiceLoss and LovaszLoss [0,1]
            self.LovaszLoss(outputs, targets)
        )


