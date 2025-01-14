from omegaconf import DictConfig
from hydra.utils import instantiate


import hydra
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    import shutil
    shutil.rmtree("saved_models", ignore_errors=True)

    ## 1. Parse config & get experiment output dir
    from omegaconf import OmegaConf
    str_config = OmegaConf.to_yaml(cfg)
    # print(str_config)

    
    ## 2. Prepare your dataset
    criterion = instantiate(cfg.criterion)

    # 2.2 Load the data
    dataloaders = instantiate(cfg.dataloaders)

    ## 3. Define your clients ##

    # 3.1 Configarations
    epochs = cfg.client_config.local_epochs
    lr = cfg.client_config.lr
    step_size = cfg.client_config.step_size
    gamma = cfg.client_config.gamma
    save_dir = cfg.client_config.save_dir

    # 3.2 torch instantiatation
    import torch
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.criterion)
    metrics = instantiate(cfg.metrics)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # 3.3 Clients generataion
    from client import generate_client_fn
    client_fn = generate_client_fn(dataloaders,
                                    model,
                                    epochs,
                                    criterion,
                                    metrics,
                                    optimizer,
                                    scheduler,
                                    save_dir,
                                    )


    ## 4. Define the strategy
    strategy = instantiate(cfg.strategy)


    ## 5. Start Simulation
    import flwr as fl
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )
    
    ## 6. Save the results
    from hydra.core.hydra_config import HydraConfig
    save_path = HydraConfig.get().runtime.output_dir

    import os
    save_path = os.path.join(save_path, "results.json")

    results = {
        "history" : history
    }

    import pickle
    with open(save_path, "wb") as file:
        pickle.dump(results, file , protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()