import torch
from torch import nn

# Mean IoU metric
class MeanIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def IoU_coef(self, y_pred, y_true, smooth=0.0001):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        total = torch.sum(y_true_f + y_pred_f)
        union = total - intersection

        return (intersection + smooth)/(union + smooth)

    def Mean_IoU(self, y_pred, y_true, NUM_OF_CLASSES=6, smooth=0.0001):
        IoU_Score=0

        for index in range(NUM_OF_CLASSES):
            IoU_Score += self.IoU_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 1)

        return IoU_Score/NUM_OF_CLASSES

    def forward(self, y_pred, y_true):
        return self.Mean_IoU(y_pred, y_true)
