import torch
import segmentation_models_pytorch.losses as losses

# Initialize DiceLoss for multilabel mode
dice_loss = losses.DiceLoss(
    mode="multilabel",  # Since y_true is one-hot encoded
    from_logits=True,   # Your model outputs raw logits
    smooth=1.0,         # Smooth term to avoid division by zero
    ignore_index=None,  # Use None if no pixels should be ignored
    log_loss=False      # Set True if you want to compute -log(dice_coeff) instead of 1 - dice_coeff
)

# Example tensors
y_pred = torch.randn(1, 6, 192, 192, requires_grad=True)  # Raw logits
y_true = torch.randint(0, 2, (1, 6, 192, 192)).float()    # One-hot encoded ground truth

# Compute Dice loss
loss = dice_loss(y_pred, y_true)

# Backward pass
loss.backward()

print("shape:", y_true.shape)
