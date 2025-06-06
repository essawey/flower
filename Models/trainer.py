import time
from collections import defaultdict
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import trange
import wandb
from .utils import save_model, plot_curve
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", message="Seems like `optimizer.step()` has been overridden after learning rate scheduler initialization")




class Trainer:
    """
    A generic training class for PyTorch models with wandb integration for federated learning.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        metrics: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        save_dir: str,
        device: torch.device,
        client_id: str,
    ):
        """
        Initialize the Trainer object.
        
        Args:
            model: The PyTorch model to be trained.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            epochs: Total number of training epochs.
            metrics: Module to calculate metrics.
            criterion: Loss function.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
            save_dir: Directory to save model checkpoints.
            device: Device to run training on (e.g., 'cuda' or 'cpu').
            client_id: Identifier for the client (useful in federated learning scenarios).
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.metrics = metrics
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device
        self.client_id = client_id
        self.epoch = 0  # To track the current epoch during training


    def train_model(self, config) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            A dictionary containing metrics tracked during training.
        """

        import wandb
        current_round = config['current_round']
        client_id = self.client_id
        run_name = fr"round_{current_round}_client_{client_id}"

        config = OmegaConf.to_container(config, resolve=True)

        wandb.init(name=run_name, config=config)


        wandb.log({"Round": int(current_round), "Client": int(client_id)})



        metrics_list = defaultdict(list)
        start_time = time.time()

        # Progress bar for visualizing epoch progress
        progressbar = trange(self.epochs, desc="Training")
        
        for _ in progressbar:
            self.epoch += 1  # Increment epoch counter
            self.model.train()  # Set the model to training mode
            # wandb.watch(self.model, log="all", log_freq=10)
            epoch_metrics = defaultdict(float)  # Metrics accumulator for the current epoch
            num_batches = 0  # To calculate batch-wise average metrics

            for x, y in self.train_dataloader:
                # Move inputs and targets to the specified device
                inputs, targets = x.to(self.device), y.to(self.device)

                # Zero gradients to prevent accumulation
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)

                # Backward pass and parameter update
                loss.backward()
                self.optimizer.step()

                # Compute metrics for the current batch
                batch_metrics = self.metrics(outputs, targets)
                batch_metrics.update(self.criterion.get_losses(outputs, targets))  # Additional losses
                batch_metrics["loss"] = loss.item()

                # Accumulate batch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

            
            wandb.log({"epoch": self.epoch})
            # Average metrics over all batches
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= num_batches
            # Store epoch metrics for tracking
            for key, value in epoch_metrics.items():
                wandb.log({key: value})
                metrics_list[key].append(value)

            # progressbar.set_description(
            #     f"Client {self.client_id} |",
            #     f"Epoch {self.epoch} |",
            #     f"Loss: {epoch_metrics['loss']:.3f} |",
            #     f"IoU: {epoch_metrics.get('iou_score_globally'):.3f}",
            # )

        # Training complete
        time_elapsed = time.time() - start_time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        save_model(self.model, self.save_dir, self.client_id, current_round)
        plot_curve(metrics_list, self.save_dir, self.client_id, current_round)
        wandb.finish()
        return dict(metrics_list)

    def val_model(self) -> Dict[str, List[float]]:
        """
        Validation Mode
        """
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        

        metrics_list = defaultdict(float)
        num_batches = 0  # To calculate batch-wise average metrics


        # Disable gradient computation during validation to save memory and computation
        with torch.no_grad():
            for x, y in self.val_dataloader:
                # Send inputs and targets to the device (GPU or CPU)
                inputs, targets = x.to(self.device), y.to(self.device)

                # Forward pass (no backward pass here)
                outputs = self.model(inputs)

                # Compute the loss
                loss = self.criterion(outputs, targets)

                # Compute the metrics for the current batch
                batch_metrics = self.metrics(outputs, targets)
                batch_metrics.update(self.criterion.get_losses(outputs, targets))
                batch_metrics["loss"] = loss.item()

                # Accumulate batch metrics
                for key, value in batch_metrics.items():
                    metrics_list[key] += value
                num_batches += 1

        # Average metrics over all batches
        for key in metrics_list.keys():
            metrics_list[key] /= num_batches

        print(dict(metrics_list))
        return dict(metrics_list)
