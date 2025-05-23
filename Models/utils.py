import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import os
import json

def save_model(model: torch.nn.Module, save_dir: str, client_id: str, current_round):

    # Base model name: round R, client C @ last epoch
    model.save_pretrained(
        save_directory=f"round_{current_round}_client_{client_id}",
        push_to_hub=False
    )

def plot_curve(metrics_list: Dict[str, List[float]], save_dir, client_id, current_round):
    # Separate losses and scores (simple heuristic by keywords)
    loss_keys = [k for k in metrics_list if 'loss' in k.lower()]
    score_keys = [k for k in metrics_list if k not in loss_keys]
    # save the metrics_list in a file

    with open(f"metrics_round_{current_round}_client_{client_id}.json", "w") as f:
        json.dump(dict(metrics_list), f, indent=4)

    plt.figure(figsize=(18, 8))

    # Plot Scores
    plt.subplot(1, 2, 1)
    colors = plt.get_cmap('tab20').colors
    for i, key in enumerate(score_keys):
        values = metrics_list[key]
        plt.plot(range(len(values)), values, label=key, color=colors[i % 20], marker='o', markersize=4)
    plt.title('Metrics Scores Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.grid(True)
    # plt.legend(fontsize=8)

    # Plot Losses
    plt.subplot(1, 2, 2)
    for i, key in enumerate(loss_keys):
        values = metrics_list[key]
        plt.plot(range(len(values)), values, label=key, color=colors[i % 20], marker='o', markersize=4)
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    # plt.legend(fontsize=8)

    plt.suptitle(f'Training Metrics - Round {current_round} - Client {client_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"plot_round_{current_round}_client_{client_id}.png")
    plt.close()
