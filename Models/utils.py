import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import os


def save_model(model: torch.nn.Module, save_dir: str, client_id: str, current_round):

    # Base model name: round R, client C @ last epoch
    model.save_pretrained(
        save_directory=f"round_{current_round}_client_{client_id}",
        push_to_hub=False
    )

def plot_curve(metrics_list: Dict[str, List[float]], save_dir, client_id, current_round):
    
    plt.figure(figsize=(12, 8))
    for key, values in metrics_list.items():
        plt.plot(values, label=key)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_round_{current_round}_client_{client_id}.png")