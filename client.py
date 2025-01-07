from collections import OrderedDict
from Models import Trainer
import torch
import flwr as fl
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, 
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                model: torch.nn.Module,
                epochs: int,
                criterion: torch.nn.Module,
                metric: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: lr_scheduler._LRScheduler,
                save_dir: str,
                ) -> None:
        super().__init__()
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the trainer instance
        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model=self.model,
            epochs=self.epochs,
            metric=self.metric,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_dir=self.save_dir,
            device=self.device,
        )
    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        # BatchNorm layers have a shape of torch.Size([0])
        # We need to convert them to torch.Size([1]) to load the model
        for k, v in state_dict.items():
            if 'num_batches_tracked' in k and v.shape == torch.Size([0]):
                state_dict[k] = torch.tensor(0)

        self.model.load_state_dict(state_dict)

    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        results = self.trainer.train_model()

        train_loss = results["train_loss"][-1]
        train_iou = results["train_iou"][-1]

        return self.get_parameters({}), len(self.train_dataloader), {"Train Loss": train_loss,"train_iou": train_iou}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)


        results = self.trainer.val_model()

        avg_loss = results["val_loss"][-1]
        val_iou = results["val_iou"][-1]

        return float(avg_loss), len(self.val_dataloader), {"Average IoU": val_iou}


def generate_client_fn(dataloaders: dict,
                        model: torch.nn.Module,
                        epochs: int,
                        criterion: torch.nn.Module,
                        metric: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: lr_scheduler._LRScheduler,
                        save_dir: str,
                        ):

    train_dataloader = dataloaders['Train']
    val_dataloader = dataloaders['Validation']

    def client_fn(cid: str):
        return FlowerClient(
            train_dataloader = train_dataloader[int(cid)],
            val_dataloader = val_dataloader[int(cid)],
            model = model,
            epochs = epochs,
            criterion = criterion,
            metric = metric,
            optimizer = optimizer,
            scheduler = scheduler,
            save_dir = save_dir,
        )

    return client_fn