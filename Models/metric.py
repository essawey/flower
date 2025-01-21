import torch
from torch import nn
import torch.nn.functional as F

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

    def forward(self, outputs, targets):
        return self.Mean_IoU(outputs, targets)

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

    def forward(self, outputs, targets):
        return self.Mean_Dice(outputs, targets)


from segmentation_models_pytorch import metrics 

class Metrics(nn.Module):
    # https://chatgpt.com/share/6783ade3-3008-8010-b47e-0a9c98293063
    # reduction = weighted 
    # 1. Looks at the entire dataset as a whole.
    # 2. Assigns a weight to each class based on how many times it appears across all images.
    # 3. Then calculates a single score for the dataset, ensuring rare classes have fair representation.

    # reduction = weighted-imagewise
    # 1. First looks at each image individually.
    # 2. Within each image, assigns weights to classes based on their frequency within that specific image.
    # 3. Then calculates a score for each image and averages these scores across all images.

    # Beta > 1: Recall is given more weight than precision.
    # Beta < 1: Precision is given more weight than recall.

    def __init__(self, mode, threshold, ignore_index, num_classes, smooth):
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.smooth = smooth

        self.class_weights = [
            9.4507010467442780, # Neoplastic cells
            56.859724293563640, # Inflammatory cells
            26.286454373302785, # Connective/Soft tissue cells
            1474.9347269860261, # Dead cells
            39.952413962033800, # Epithelial cells
            1.2302386418184008, # Background
        ]
        self.mean_iou = MeanIoU(smooth=self.smooth, num_classes=self.num_classes)
        self.mean_dice = MeanDice(smooth=self.smooth, num_classes=self.num_classes)

    def compute_stats(self, outputs, targets):
        """Compute true positives, false positives, false negatives, and true negatives."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        targets = targets.to(device)
        outputs = outputs.to(device)

        targets = torch.argmax(targets, dim=1)
        outputs = torch.argmax(outputs, dim=1)

        targets = targets.int()
        targets = torch.where(targets == self.ignore_index, torch.tensor(-1, device=device), targets)
        tp, fp, fn, tn = metrics.get_stats(outputs, targets, mode="multiclass", ignore_index=-1, num_classes = self.num_classes)
        return tp, fp, fn, tn

    def forward(self, outputs, targets):
        """Compute all metrics and return as a dictionary."""

        build_in_metrics = {
        "mean_iou": self.mean_iou(outputs, targets).item(),
        "mean_dice": self.mean_dice(outputs, targets).item(),
        }

        tp, fp, fn, tn = self.compute_stats(outputs, targets)

        imported_metrics = {
            "iou_score_globally": metrics.iou_score(tp, fp, fn, tn, reduction='weighted', class_weights=self.class_weights).item(),
            "f1_score_globally": metrics.f1_score(tp, fp, fn, tn, reduction='weighted', class_weights=self.class_weights).item(),
            "balanced_accuracy_globally": metrics.balanced_accuracy(tp, fp, fn, tn, reduction='weighted', class_weights=self.class_weights).item(),
            "precision_globally": metrics.precision(tp, fp, fn, tn, reduction='weighted', class_weights=self.class_weights).item(),
            "recall_globally": metrics.recall(tp, fp, fn, tn, reduction='weighted', class_weights=self.class_weights).item(),
            "f_precision_globally": metrics.fbeta_score(tp, fp, fn, tn, beta=0.5, reduction='weighted', class_weights=self.class_weights).item(),
            "f_recall_globally": metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction='weighted', class_weights=self.class_weights).item(),

            "iou_score_imagewise": metrics.iou_score(tp, fp, fn, tn, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "f1_score_imagewise": metrics.f1_score(tp, fp, fn, tn, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "balanced_accuracy_imagewise": metrics.balanced_accuracy(tp, fp, fn, tn, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "precision_globally": metrics.precision(tp, fp, fn, tn, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "recall_imagewise": metrics.recall(tp, fp, fn, tn, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "f_precision_imagewise": metrics.fbeta_score(tp, fp, fn, tn, beta=0.5, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
            "f_recall_imagewise": metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction='weighted-imagewise', class_weights=self.class_weights).item(),
        }
        imported_metrics.update(build_in_metrics)

        return imported_metrics

