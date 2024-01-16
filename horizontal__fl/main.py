
import pickle
from pathlib import Path

import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

# from dataset import prepare_dataset
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    # trtainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients,
                                                                #    cfg.batch_size)
    csv = '/home/pavel/olina/research/data/part-00016-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'

    trtainloaders, validationloaders, testloader = prepare_dataset( csv, 
                                                                    mode=cfg.mode_data, #'STRATIFIED' 'SINGLE_ATTACK' 'RANDOM' 'EXCLUDE_SINGLE_ATTACK' 'HALF_BENIGN'
                                                                    normalization=cfg.normalization, #'StandardScaler',
                                                                    feature_extractor=cfg.feature_extractor,
                                                                    num_partitions=cfg.num_clients,
                                                                    max_partitions_size=cfg.max_partitions_size,
                                                                    batch_size=cfg.batch_size,
                                                                    val_ratio=cfg.val_ratio
                                                                    )
    print(len(trtainloaders))
    ## 3. Define your clients
    client_fn = generate_client_fn(trtainloaders, validationloaders, cfg.model)

    ## 4. Define your strategy
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=get_evaluate_fn(cfg.model, testloader)
                           )

    ## 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 4,
            "num_gpus": 0.125,
        },  
    )
    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pk'
    results = {'history': history, 'anythingelse': "here"}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    main()