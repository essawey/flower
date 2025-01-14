import time
from collections import defaultdict
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import trange
from .utils import save_model, plot_curve

import warnings

warnings.filterwarnings("ignore", message="Seems like `optimizer.step()` has been overridden after learning rate scheduler initialization")


class Trainer:
    """
    A generic training class for PyTorch models.
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

    def train_model(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            A dictionary containing metrics tracked during training.
        """
        # Initialize a dictionary to store metrics, using defaultdict for convenience
        metrics_list = defaultdict(list)

        # Start tracking training time
        start_time = time.time()

        # Progress bar for visualizing epoch progress
        progressbar = trange(self.epochs, desc="Training")
        
        for _ in progressbar:
            self.epoch += 1  # Increment epoch counter
            self.model.train()  # Set the model to training mode

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
                
                # Perform a step of the learning rate scheduler
                # self.scheduler.step()

                # Compute metrics for the current batch
                batch_metrics = self.metrics(outputs, targets)
                batch_metrics.update(self.criterion.get_losses(outputs, targets))  # Additional losses
                batch_metrics["loss"] = loss.item()

                # Accumulate batch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1


            # Average metrics over all batches
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= num_batches

            # Store epoch metrics for tracking
            for key, value in epoch_metrics.items():
                metrics_list[key].append(value)

            
            progressbar.set_description(
                f'''
                Client {self.client_id} | Epoch {self.epoch} | Loss: {epoch_metrics['loss']:.3f} | IoU: {epoch_metrics.get('iou_score_globally'):.3f}
                '''
            )

            # Save model checkpoint
            # self.save_model()

        # Training complete
        time_elapsed = time.time() - start_time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        # self.plot_curve(metrics_list)
        
        return dict(metrics_list)


    def val_model(self) -> Dict[str, List[float]]:
        """
        Validation Mode
        """
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        
        metrics_list = defaultdict(list)  # Initialize the dictionary to store validation metrics
        epoch_metrics = defaultdict(float)  # Metrics accumulator for the current epoch
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
                batch_metrics.update(self.criterion.get_losses(outputs, targets))  # Add additional losses if any
                batch_metrics["loss"] = loss.item()

                # Accumulate batch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

        # Average metrics over all batches
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches

        # Store epoch metrics for tracking
        for key, value in epoch_metrics.items():
            metrics_list[key].append(value)

        return dict(epoch_metrics)
