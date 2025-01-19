import wandb
import time

def train_one_client(client_id):
    # You can create a new run for each client.
    # Use `reinit=True` so that you can initialize multiple W&B runs in the same script.
    run = wandb.init(
        project="my-project-o1",
        name=f"client_{client_id}-run",
        reinit=True
    )

    for epoch in range(3):
        loss = 0.1 * epoch * (client_id + 1)  # Just a dummy formula
        wandb.log({"epoch": epoch, "loss": loss})
        time.sleep(0.5)

def main():
    # Suppose you have 4 clients
    for client_id in range(4):
        train_one_client(client_id)

if __name__ == "__main__":
    main()

# import wandb
# import time

# wandb.init(project="my-project", name="multi-client-one-run")

# def train_one_client(client_id):
#     for epoch in range(3):
#         loss = 0.1 * epoch * (client_id + 1)  # Some dummy formula
#         wandb.log(
#             {
#                 f"loss_client_{client_id}": loss
#             }
#         )
#         time.sleep(0.5)

# def main():
#     # 4 clients
#     for client_id in range(4):
#         train_one_client(client_id)

#     # When everything is done, finish
#     wandb.finish()

# if __name__ == "__main__":
#     main()


# from collections import OrderedDict
# from Models import Trainer
# import torch
# import flwr as fl
# from torch.utils.data import DataLoader
# import torch.optim.lr_scheduler as lr_scheduler
# import wandb  # Import wandb

# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, 
#                 train_dataloader: DataLoader,
#                 val_dataloader: DataLoader,
#                 model: torch.nn.Module,
#                 epochs: int,
#                 criterion: torch.nn.Module,
#                 metrics: torch.nn.Module,
#                 optimizer: torch.optim.Optimizer,
#                 scheduler: lr_scheduler._LRScheduler,
#                 save_dir: str,
#                 client_id: str
#                 ) -> None:
#         super().__init__()
        
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.model = model
#         self.epochs = epochs
#         self.criterion = criterion
#         self.metrics = metrics
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.save_dir = save_dir
#         self.client_id = client_id
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize the trainer instance
#         self.trainer = Trainer(
#             train_dataloader=self.train_dataloader,
#             val_dataloader=self.val_dataloader,
#             model=self.model,
#             epochs=self.epochs,
#             metrics=self.metrics,
#             criterion=self.criterion,
#             optimizer=self.optimizer,
#             scheduler=self.scheduler,
#             save_dir=self.save_dir,
#             device=self.device,
#             client_id = self.client_id
#         )
        
#         # Initialize a run counter to track federated learning rounds
#         self.run_counter = 0  # Optional: Only if 'round' is not in config

#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

#         # Handle BatchNorm layers if necessary
#         for k, v in state_dict.items():
#             if 'num_batches_tracked' in k and v.shape == torch.Size([0]):
#                 state_dict[k] = torch.tensor(0)

#         self.model.load_state_dict(state_dict)

#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
#     def fit(self, parameters, config):
#         """
#         Trains the model on the client's local data.
#         """
#         # Set the model parameters from the server
#         self.set_parameters(parameters)
        
#         # Extract the current federated learning round from config
#         current_round = config.get("round", self.run_counter)
        
#         # Increment run counter if 'round' is not provided
#         if "round" not in config:
#             self.run_counter += 1
#             current_round = self.run_counter
        
#         # Initialize a new W&B run for this round
#         wandb.init(
#             project="o1-test",  # Your W&B project name
#             name=f"client_{self.client_id}_round_{current_round}",  # Unique run name
#             reinit=True,  # Allow multiple runs in the same script
#             config={
#                 "client_id": self.client_id,
#                 "round": current_round,
#                 "epochs": self.epochs,
#                 "optimizer": type(self.optimizer).__name__,
#                 "scheduler": type(self.scheduler).__name__,
#                 # Add other hyperparameters as needed
#             }
#         )
        
#         # Watch the model to log gradients and parameters
#         wandb.watch(self.model, log="all", log_freq=10)
        
#         # Train the model
#         metrics_dict = self.trainer.train_model(config)
        
#         # Log training metrics to W&B
#         wandb.log({"train_metrics": metrics_dict, "round": current_round})
        
#         # Optionally, you can save model artifacts
#         # wandb.save(os.path.join(self.save_dir, f"model_client_{self.client_id}_round_{current_round}.pt"))
        
#         # Return updated parameters, number of training examples, and metrics
#         return self.get_parameters({}), len(self.train_dataloader.dataset), metrics_dict

#     def evaluate(self, parameters, config):
#         """
#         Evaluates the model on the client's local validation data.
#         """
#         # Set the model parameters from the server
#         self.set_parameters(parameters)
        
#         # Extract the current federated learning round from config
#         current_round = config.get("round", self.run_counter)
        
#         # Evaluate the model
#         results = self.trainer.val_model()
        
#         # Log validation metrics to W&B
#         wandb.log({"val_metrics": results, "round": current_round})
        
#         # Optionally, finish the W&B run after evaluation
#         wandb.finish()
        
#         # Return loss, number of validation examples, and results
#         return results['loss'], len(self.val_dataloader.dataset), results
