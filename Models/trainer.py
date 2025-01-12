import time
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from .utils import save_model, plot_curve
from hydra.utils import instantiate


class Trainer:

    def __init__(self,
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

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epoch = 0
        self.epochs = epochs
        self.metrics = metrics
        # self.loss = criterion
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device
        self.client_id = client_id

        self.metrics_dict = {

            ######## TRAINING KEYS ########

            # Loss
            "train_loss": [],

            # Global metrics
            "train_iou_score_globally": [],
            "train_f1_score_globally": [],
            "train_accuracy_globally": [],
            "train_recall_globally": [],
            "train_f_precision_globally": [],
            "train_f_recall_globally": [],

            # Image wise metrics
            "train_iou_score_imagewise": [],
            "train_f1_score_imagewise": [],
            "train_accuracy_imagewise": [],
            "train_recall_imagewise": [],
            "train_f_precision_imagewise": [],
            "train_f_recall_imagewise": [],

            # built-in 
            "train_MeanIoU": [],
            "train_MeanDice": [],

            ######## VALIDATION KEYS ########

            "val_loss": [],

            "val_iou_score_globally": [],
            "val_f1_score_globally": [],
            "val_accuracy_globally": [],
            "val_recall_globally": [],
            "val_f_precision_globally": [],
            "val_f_recall_globally": [],

            "val_iou_score_imagewise": [],
            "val_f1_score_imagewise": [],
            "val_accuracy_imagewise": [],
            "val_recall_imagewise": [],
            "val_f_precision_imagewise": [],
            "val_f_recall_imagewise": [],

            "val_MeanIoU": [],
            "val_MeanDice": [],

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

            metrics_dict = {
                "train_loss": [],
                "val_loss": [],

                "train_MeanIoU": [],
                "val_MeanIoU": [],
                "train_MeanDice": [],
                "val_MeanDice": [],

                "iou_score_globally": [],
                "f1_score_globally": [],
                "accuracy_globally": [],
                "recall_globally": [],
                "f_precision_globally": [],
                "f_recall_globally": [],

                "iou_score_imagewise": [],
                "f1_score_imagewise": [],
                "accuracy_imagewise": [],
                "recall_imagewise": [],
                "f_precision_imagewise": [],
                "f_recall_imagewise": [],
            }

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

                # Backward pass
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                ##########################
                ### Metrics Calculation ##
                ##########################
                loss_value = loss.item()

                metrics_dict = self.metrics(outputs, targets)
                losses_dict = self.criterion.get_losses(outputs, targets)
                print(":"*10)
                print(metrics_dict)
                print("*"*10)
                print(losses_dict)
                # running_losses.append(loss_value)

                # # Calculate the iou
                # iou_value = iou.item()
                # running_ious.append(iou_value)

                # # Calculate the dice
                # dice_value = dice.item()
                # running_dices.append(dice_value)

                # running_metrics["iou"].append(metrics_global["iou_score"].item())
                # running_metrics["dice"].append(metrics_global["f1_score"].item())
                # running_metrics["accuracy"].append(metrics_global["accuracy"].item())
                # running_metrics["f1"].append(metrics_global["f1_score"].item())



            # self.scheduler.step()
            # print("-"*20)
            # print(self.client_id)
            # print(self.results)
            # print("-"*20)

            # self.results["train_loss"].append(np.mean(running_losses))
            # self.results["train_iou"].append(np.mean(running_ious))
            # self.results["train_dice"].append(np.mean(running_dices))

            # updateText = f'''
            # Local Training for client {self.client_id}
            # Epoch: {self.epoch}
            # Train loss: {self.results['train_loss'][-1]:.3f}
            # Train IoU: {self.results['train_iou'][-1]:.3f}
            # '''

            # progressbar.set_description(updateText)        
            # print(updateText)

            # Save checkpoints every epochs
        #     save_model(self.model, self.save_dir, self.client_id)

        # time_elapsed = time.time() - start_time
        # print('\n')
        # print('-' * 20)
        # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        # plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def val_model(self):
        """
        Validation Mode
        """
        self.model.to(self.device)
        self.model.eval() # Validation mode
        # running_dices, running_ious, running_losses = [], [], []

        for x, y in self.val_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                # loss_value = loss.item()
                # running_losses.append(loss_value)

                # # Calculate the iou
                # iou = self.MeanIoU(outputs, targets)
                # iou_value = iou.item()
                # running_ious.append(iou_value)

                # # Calculate the dice
                # dice = self.MeanDice(outputs, targets)
                # dice_value = dice.item()
                # running_dices.append(dice_value)


        # self.results["val_loss"].append(np.mean(running_losses))
        # self.results["val_iou"].append(np.mean(running_ious))
        # self.results["val_dice"].append(np.mean(running_dices))


        # print(
        #     f'''
        #     LOACL MODEL VALIDATION LOSS for {self.client_id}
        #     Validation loss: {self.results['val_loss'][-1]:.3f}
        #     Validation IoU: {self.results['val_iou'][-1]:.3f}
        #     Validation Dice: {self.results['val_dice'][-1]:.3f}
        #     '''
        # )
        
        # return self.results
        return loss