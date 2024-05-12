import pickle
from pathlib import Path

import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn, weighted_average
from server import get_on_fit_config, get_evaluate_fn
# from monitor import monitor
import ray 

# Инициализация Ray с использованием адреса "auto"

@hydra.main(config_path="conf", config_name="base_federated", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # monitor()
    ## 2. Prepare your dataset
    csv = '/home/pavel/olina/research/neural_network_fl/benighn_0.csv'
    trtainloaders, validationloaders, testloader = prepare_dataset( csv, 
                                                                    mode=cfg.mode_data, #'STRATIFIED' 'SINGLE_ATTACK' 'RANDOM' 'EXCLUDE_SINGLE_ATTACK' 'HALF_BENIGN'
                                                                    type_normalization=cfg.type_normalization, #'Standard',
                                                                    type_feature_extractor=cfg.type_feature_extractor,
                                                                    num_partitions=cfg.num_clients,
                                                                    max_partitions_size=cfg.max_partitions_size,
                                                                    batch_size=cfg.batch_size,
                                                                    val_ratio=cfg.val_ratio
                                                                    )
    # print(len(trtainloaders))

    ## 3. Define your clients
    client_fn = generate_client_fn(trtainloaders, validationloaders, cfg.model)

    ## 4. Define your strategy

    # Total resources for simulation
    num_cpus = 4
    num_gpus = 1
    ram_memory = 16_000 * 1024 * 1024 # 16 GB

    # Single client resources
    client_num_cpus = 0.5
    client_num_gpus = 0

    # ray.init(dashboard_host="93.81.242.150")

    strategy = instantiate(cfg.strategy,
                           evaluate_fn=get_evaluate_fn(cfg.model, testloader),
                           evaluate_metrics_aggregation_fn=weighted_average,
                           )

    ## 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
      
        strategy=strategy,  # our strategy of choice
        ray_init_args = {
            "include_dashboard": True, # we need this one for tracking
            # "num_cpus": num_cpus,
            # "num_gpus": num_gpus,
            # "memory": ram_memory,
    },
        client_resources={
            "num_cpus": client_num_cpus, # указывает на количество ядер процессора, которые получит клиент
            "num_gpus": client_num_gpus, # указывает на соотношение памяти графического процессора, назначенного клиенту
        },  
    )
    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pk'
    results = {
                'history': history, 
              }

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    main()