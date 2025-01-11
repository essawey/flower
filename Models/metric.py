import torch
from torch import nn

# Mean IoU metric
class MeanIoU(nn.Module):
    def __init__(self, smooth, num_classes):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def IoU_coef(self, y_pred, y_true):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        total = torch.sum(y_true_f + y_pred_f)
        union = total - intersection
        return (intersection + self.smooth)/(union + self.smooth)

    def Mean_IoU(self, y_pred, y_true):
        IoU_Score=0
        for index in range(self.num_classes):
            IoU_Score += self.IoU_coef(y_true[:,index,:,:], y_pred[:,index,:,:])
        return IoU_Score/self.num_classes

    def forward(self, y_pred, y_true):
        return self.Mean_IoU(y_pred, y_true)

# Mean Dice metric
class MeanDice(nn.Module):
    def __init__(self, smooth, num_classes):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes


    def dice_coef(self, y_pred, y_true):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def Mean_Dice(self, y_pred, y_true):
        Dice_Score=0
        for index in range(self.num_classes):
            Dice_Score += self.dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:])
        return Dice_Score/self.num_classes

    def forward(self, y_pred, y_true):
        return self.Mean_Dice(y_pred, y_true)

