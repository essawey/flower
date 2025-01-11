
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os


# Save the model to the target dir
def save_model(model: torch.nn.Module, target_dir: str, client_id: str):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory and model save path
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)


    # only call this code if the path exists
    folders = [folder for folder in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, folder))]
    roundsSaved = [int(re.search(r'\d', text).group()) if re.search(r'\d', text) else 1 for text in folders] + [1]

    if os.path.exists(os.path.join(target_dir, f"model_round_{1}_client_{client_id}")):
        current_round = max(roundsSaved) + 1
    else:
        current_round = max(roundsSaved)
    model_save_path = target_dir_path / f"model_round_{current_round}_client_{client_id}"

    # Save the model using segmentation_models_pytorch utility
    model.save_pretrained(save_directory=model_save_path, push_to_hub=False)

# Plot the training curve
def plot_curve(results: dict, epochs: int):

    train_ious = np.array(results["train_iou"])
    train_dices = np.array(results["train_dice"])
    train_losses = np.array(results["train_loss"])

    plt.plot(np.arange(0, epochs, 1), train_losses, label='Train loss')
    plt.plot(np.arange(0, epochs, 1), train_ious, label='Train IoU')
    plt.plot(np.arange(0, epochs, 1), train_dices, label='Train Dice')

    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig("FIXME.png")
