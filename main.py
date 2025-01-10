import hydra
from omegaconf import DictConfig
from client import generate_client_fn
import torch
import flwr as fl
from server import get_on_fit_config, get_evaluate_fn
from hydra.utils import instantiate


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    # print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset

    # 2.2 Load the data
    dataloaders = instantiate(cfg.dataloaders)

    ## 3. Define your clients

    # 3.1 Configarations
    epochs = cfg.client_config.local_epochs
    lr = cfg.client_config.lr
    step_size = cfg.client_config.step_size
    gamma = cfg.client_config.gamma
    save_dir = cfg.client_config.save_dir

    # 3.2 Models
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.criterion)

    metric = instantiate(cfg.metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 3.3 Clients
    client_fn =  generate_client_fn(dataloaders,
                                    model,
                                    epochs,
                                    criterion,
                                    metric,
                                    optimizer,
                                    scheduler,
                                    save_dir,
                                    )

    print(f"Number of clients main : {cfg.num_clients}")

    ## 4. Define your strategy
    strategy = instantiate(cfg.strategy)
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.00001,
    #     min_fit_clients=cfg.num_clients_per_round_fit,
    #     fraction_evaluate=0.00001,
    #     min_evaluate_clients=cfg.num_clients_per_round_eval,
    #     min_available_clients=cfg.num_clients,
    #     on_fit_config_fn=get_on_fit_config(cfg=cfg),
    #     evaluate_fn=get_evaluate_fn(model,
    #                                 dataloaders,
    #                                 criterion,
    #                                 metric,
    #                                 ),
    #     # fit_metrics_aggregation_fn=
    # )

    print(f"Number of clients : {cfg.num_rounds}")

    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )
    ## 6. Save your results
    from hydra.core.hydra_config import HydraConfig
    import os
    import pickle

    save_path = HydraConfig.get().runtime.output_dir
    save_path = os.path.join(save_path, "results.json")

    results = {
        "history" : history
    }

    with open(save_path, "wb") as file:
        pickle.dump(results, file , protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()