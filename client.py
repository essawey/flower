from collections import OrderedDict
from Models import Trainer
import torch
import flwr as fl
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import wandb
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, 
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                model: torch.nn.Module,
                epochs: int,
                criterion: torch.nn.Module,
                metrics: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: lr_scheduler._LRScheduler,
                save_dir: str,
                client_id: str
                ) -> None:
        super().__init__()
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the trainer instance
        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model=self.model,
            epochs=self.epochs,
            metrics=self.metrics,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_dir=self.save_dir,
            device=self.device,
            client_id = self.client_id
        )
        
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics_dict = self.trainer.train_model(config)
        return self.get_parameters(config), len(self.train_dataloader), metrics_dict


    def evaluate(self, parameters, config):

        current_round = config.get("current_round", 0)
        self.set_parameters(parameters)
        results = self.trainer.val_model()
        from pprint import pprint
        pprint(results)
        return results['loss'], len(self.val_dataloader), results


def generate_client_fn(dataloaders: dict,
                        model: torch.nn.Module,
                        epochs: int,
                        criterion: torch.nn.Module,
                        metrics: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: lr_scheduler._LRScheduler,
                        save_dir: str,
                        ):

    def client_fn(cid: str):
        
        train_dataloader = dataloaders['Train']
        val_dataloader = dataloaders['Validation']

        return FlowerClient(
            train_dataloader = train_dataloader[int(cid)],
            val_dataloader = val_dataloader[int(cid)],
            model = model,
            epochs = epochs,
            criterion = criterion,
            metrics = metrics,
            optimizer = optimizer,
            scheduler = scheduler,
            save_dir = save_dir,
            client_id = cid
        )

    return client_fn