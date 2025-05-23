# import json
# from logging import INFO

# import torch
# import wandb

# from collections import OrderedDict
# from pytorch_example.task import Net, create_run_dir, set_weights

# from flwr.common import logger, parameters_to_ndarrays
# from flwr.common.typing import UserConfig
# from flwr.server.strategy import FedAvg

# PROJECT_NAME = "FLOWER-advanced-pytorch"


# from hydra.utils import instantiate



# class CustomFedAvg(FedAvg):
#     """A class that behaves like FedAvg but has extra functionality.

#     This strategy: (1) saves results to the filesystem, (2) saves a
#     checkpoint of the global  model when a new best is found, (3) logs
#     results to W&B if enabled.
#     """

#     def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Create a directory where to save results from this run
#         self.save_path, self.run_dir = create_run_dir(run_config)
#         self.use_wandb = use_wandb
#         # Initialise W&B if set
#         if use_wandb:
#             self._init_wandb_project()

#         # Keep track of best acc
#         self.best_acc_so_far = 0.0

#         # A dictionary to store results as they come
#         self.results = {}

#     def _init_wandb_project(self):
#         # init W&B
#         wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

#     def _store_results(self, tag: str, results_dict):
#         """Store results in dictionary, then save as JSON."""
#         # Update results dict
#         if tag in self.results:
#             self.results[tag].append(results_dict)
#         else:
#             self.results[tag] = [results_dict]

#         # Save results to disk.
#         # Note we overwrite the same file with each call to this function.
#         # While this works, a more sophisticated approach is preferred
#         # in situations where the contents to be saved are larger.
#         with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
#             json.dump(self.results, fp)

#     def _update_best_acc(self, round, accuracy, parameters):
#         """Determines if a new best global model has been found.

#         If so, the model checkpoint is saved to disk.
#         """
#         if accuracy > self.best_acc_so_far:
#             self.best_acc_so_far = accuracy
#             logger.log(INFO, "💡 New best global model found: %f", accuracy)
#             # You could save the parameters object directly.
#             # Instead we are going to apply them to a PyTorch
#             # model and save the state dict.
#             # Converts flwr.common.Parameters to ndarrays
#             ndarrays = parameters_to_ndarrays(parameters)
#             model = instantiate(cfg.model)
#             model = set_weights(model, ndarrays)
#             # Save the PyTorch model
#             model_path = f"model_state_acc_{accuracy}_round_{round}.pth"
#             model.save_pretrained(save_directory=model_path, push_to_hub=False)

#     def store_results_and_log(self, server_round: int, tag: str, results_dict):
#         """A helper method that stores results and logs them to W&B if enabled."""
#         # Store results
#         self._store_results(
#             tag=tag,
#             results_dict={"round": server_round, **results_dict},
#         )

#         if self.use_wandb:
#             # Log centralized loss and metrics to W&B
#             wandb.log(results_dict, step=server_round)

#     def evaluate(self, server_round, parameters):
#         """Run centralized evaluation if callback was passed to strategy init."""
#         loss, metrics = super().evaluate(server_round, parameters)

#         # Save model if new best central accuracy is found
#         self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

#         # Store and log
#         self.store_results_and_log(
#             server_round=server_round,
#             tag="centralized_evaluate",
#             results_dict={"centralized_loss": loss, **metrics},
#         )
#         return loss, metrics

#     def aggregate_evaluate(self, server_round, results, failures):
#         """Aggregate results from federated evaluation."""
#         loss, metrics = super().aggregate_evaluate(server_round, results, failures)

#         # Store and log
#         self.store_results_and_log(
#             server_round=server_round,
#             tag="federated_evaluate",
#             results_dict={"federated_evaluate_loss": loss, **metrics},
#         )
#         return loss, metrics


import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask_path):
    # Define the labels
    masksLabels = [
        'Neoplastic cells',
        'Inflammatory',
        'Connective/Soft tissue cells',
        'Dead Cells',
        'Epithelial',
        'Background',
    ]
    
    # Load the mask from the provided path (assuming it's a numpy file)
    try:
        mask = np.load(mask_path)
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return

    # Ensure mask shape is (6, height, width)
    if mask.shape[0] != 6:
        if 1 in mask.shape:
            mask = np.squeeze(mask)
        mask = np.eye(6)[mask]  # One-hot encode to (height, width, 6)
        mask = np.moveaxis(mask, -1, 0)  # Rearrange to (6, height, width)

    # Create a dictionary for label indices
    labels_idx = {label_idx: label for label_idx, label in enumerate(masksLabels)}

    # Create the plot with subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # List to store the images for color bar
    channel_images = []

    # Iterate over each channel and plot the corresponding mask
    for channel_index in range(mask.shape[0]):
        max_value = np.max(mask[channel_index, :, :])
        colors = plt.cm.get_cmap('tab20', int(max_value + 1))

        # Calculate the row and column position for the subplot
        row = channel_index // 3
        col = channel_index % 3

        # Plot the mask for the current channel
        ax = axes[row, col]
        im = ax.imshow(mask[channel_index, :, :], cmap=colors, vmin=0, vmax=max_value)
        ax.set_title(f'Channel {channel_index} : {labels_idx[channel_index]}')
        ax.axis('off')
        channel_images.append(im)

    # Add a color bar for the last channel image
    cbar_ax = fig.add_axes((0.15, -0.02, 0.7, 0.03))
    fig.colorbar(channel_images[-1], cax=cbar_ax, orientation='horizontal')
    
    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

# Usage
mask_path = r"C:\Users\Essawey\Desktop\flower\PanNuke\data\Patched\Train\0\Adrenal-gland\fold1_1171_Adrenal-gland_patch3.npy"
show_mask(mask_path)
