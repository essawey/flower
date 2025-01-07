
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Save the model to the target dir
def save_model(model: torch.nn.Module, target_dir: str, epoch: int):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    check_point_name = f"model_epoch_{epoch}"
    model_save_path = target_dir_path / check_point_name

    # Save the model state_dict()
    #print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

# Plot the training curve
def plot_curve(results: dict, epochs: int):
    train_ious, val_ious = np.array(results["train_iou"]), np.array(results["val_iou"])
    train_losses, val_losses = np.array(results["train_loss"]), np.array(results["val_loss"])

    plt.plot(np.arange(0, epochs, 1), train_losses, label='Train loss')
    plt.plot(np.arange(0, epochs, 1), train_ious, label='Train IoU')
    plt.plot(np.arange(0, epochs, 1), val_losses, label='Val loss')
    plt.plot(np.arange(0, epochs, 1), val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
