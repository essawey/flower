import time
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from .utils import save_model, plot_curve

class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 epochs: int,
                 MeanIoU: torch.nn.Module,
                 MeanDice: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: lr_scheduler._LRScheduler,
                 save_dir: str,
                 device: torch.device,
                 client_id: str,
                 ):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epoch = 0
        self.epochs = epochs
        self.MeanIoU = MeanIoU
        self.MeanDice = MeanDice

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device
        self.client_id = client_id
        # Create empty results dictionary
        self.results = {
            "train_loss": [],
            "train_iou": [],
            "val_loss": [],
            "val_iou": []
        }

    def train_model(self):
        """
        Train the Model.
        calling the train and validation epochs functions as well as saving the model checkpoints.
        calling the plot_curve function.
        calculating the time taken for training and validation.
        """
        start_time = time.time()
        self.model.to(self.device)
        progressbar = trange(self.epochs, desc="Training")
        for _ in progressbar:
            # Epochs counter
            self.epoch += 1
            #progressbar.set_description(f"Epochs {self.epochs}")

            # Training block
            self.model.train() # training mode
            running_dices, running_ious, running_losses = [], [], []

            for x, y in self.train_dataloader:
                # Send to device (GPU or CPU)
                inputs = x.to(self.device)
                targets = y.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward - track history if only in train
                outputs = self.model(inputs)

                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the iou
                iou = self.MeanIoU(outputs, targets)
                iou_value = iou.item()
                running_ious.append(iou_value)

                # Calculate the dice
                dice = self.MeanDice(outputs, targets)
                dice_value = dice.item()
                running_dices.append(dice_value)

                # Backward pass
                loss.backward()
                # Update the parameters
                self.optimizer.step()

            # self.scheduler.step()
            print("-"*20)
            print(self.client_id)
            print(self.results)
            print("-"*20)

            self.results["train_loss"].append(np.mean(running_losses))
            self.results["train_iou"].append(np.mean(running_ious))
            self.results["train_dice"].append(np.mean(running_dices))

            #progressbar.set_description(f'\nTrain loss: {self.results["train_loss"][-1]} Train iou: {self.results["train_iou"][-1]}')

            print(
                f'''
                LOACL TRAINING for {self.client_id}
                Epoch: {self.epoch}
                Train loss: {self.results['train_loss'][-1]:.3f}
                Train IoU: {self.results['train_iou'][-1]:.3f}
                '''
            )

            # Save checkpoints every epochs
            save_model(self.model, self.save_dir, self.client_id)

        time_elapsed = time.time() - start_time
        print('\n')
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def val_model(self):
        """
        Validation Mode
        """
        self.model.to(self.device)
        self.model.eval() # Validation mode
        running_ious, running_losses = [], []

        for x, y in self.val_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the iou
                iou = self.metric(outputs, targets)
                iou_value = iou.item()
                running_ious.append(iou_value)

        self.results["val_loss"].append(np.mean(running_losses))
        self.results["val_iou"].append(np.mean(running_ious))


        print(
            f'''
            LOACL MODEL VALIDATION LOSS for {self.client_id}
            Validation loss: {self.results['val_loss'][-1]:.3f}
            Validation IoU: {self.results['val_iou'][-1]:.3f}
            '''
        )
        
        return self.results