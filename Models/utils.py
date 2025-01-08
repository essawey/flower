
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
from server import rounds


# Save the model to the target dir
def save_model(model: torch.nn.Module, target_dir: str, epoch: int, client_id: str):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory and model save path
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)


    # # only call this code if the path exists
    # folders = [folder for folder in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, folder))]
    # roundsSaved = [int(re.search(r'\d', text).group()) if re.search(r'\d', text) else 1 for text in folders]
    # roundsSaved.append(1)

    # current_round = max(roundsSaved) + 1

    model_save_path = target_dir_path / f"model_round_{rounds}_client_{client_id}_epoch_{epoch}"

    # Save the model using segmentation_models_pytorch utility
    model.save_pretrained(save_directory=model_save_path, push_to_hub=False)

# Plot the training curve
def plot_curve(results: dict, epochs: int):
    pass
    # train_ious = np.array(results["train_iou"])
    # train_losses = np.array(results["train_loss"])

    # plt.plot(np.arange(0, epochs, 1), train_losses, label='Train loss')
    # plt.plot(np.arange(0, epochs, 1), train_ious, label='Train IoU')
    # plt.xlabel('Epoch')
    # plt.legend(loc='upper right')
    # plt.show()
