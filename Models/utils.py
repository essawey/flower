
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_current_round_path(filename: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    # Update the filename to match the current round number
    while os.path.exists(os.path.join(save_dir, new_filename)):
        new_filename = f"{base_name}({counter}){ext}"
        counter += 1
    
    return os.path.join(save_dir, new_filename)


def save_model(model: torch.nn.Module, save_dir: str, client_id: str):
    # Base model name: round 1, client 1, last epoch
    model_path = get_current_round_path(f"model_round_{1}_client_{client_id}", save_dir)
    model.save_pretrained(save_directory=model_path, push_to_hub=False)


# Plot the training curve
from typing import Dict, List

def plot_curve(metrics_list: Dict[str, List[float]], save_dir, client_id, epoch):
    
    plt.figure(figsize=(12, 8))
    for key, values in metrics_list.items():
        plt.plot(np.arange(0, epoch, 1), values, label=key)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)

    plot_path = get_current_round_path(f"plot_round_{1}_client_{client_id}", save_dir)

    plt.savefig(f"{plot_path}.png")
