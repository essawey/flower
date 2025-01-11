import torch
from torch import nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig


# Categorical Cross Entropy Loss
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.nll_loss(y_hat.log(), y.argmax(dim=1))
    


class CombinedLoss(nn.Module):
    def __init__(self, config: DictConfig, mode=None, from_logits=None, smooth=None, ignore_index=None, log_loss=None):
        super().__init__()
        self.config = config

        # self.cross_entropy_loss = CategoricalCrossEntropyLoss()

        self.DiceLoss = config["DiceLoss"]
        self.FocalLoss = config["FocalLoss"]
        self.LovaszLoss = config["LovaszLoss"]

    def forward(self, y_pred, y_true):
        return (
            # self.cross_entropy_loss(y_pred, y_true) +
            self.DiceLoss(y_pred, y_true) + 
            self.FocalLoss(y_pred, y_true) + 
            self.LovaszLoss(y_pred, y_true)
        )











# # Multiclass Dice Loss
# class MultiDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def dice_coef(self, y_pred, y_true, smooth=0.0001):

#         y_true_f = y_true.flatten()
#         y_pred_f = y_pred.flatten()
#         intersection = torch.sum(y_true_f * y_pred_f)

#         return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

#     def dice_coef_multiclass(self, y_pred, y_true, NUM_OF_CLASSES=6, smooth=0.0001):
#         dice=0

#         for index in range(NUM_OF_CLASSES):
#             dice += self.dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 0.0001)

#         return 1 - dice/NUM_OF_CLASSES

#     def forward(self, y_pred, y_true):

#         return self.dice_coef_multiclass(y_pred, y_true)

