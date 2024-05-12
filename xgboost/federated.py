
import pickle
from pathlib import Path

import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import get_client_fn
from server import eval_config, fit_config, get_evaluate_fn, evaluate_metrics_aggregation


@hydra.main(config_path="conf", config_name="base_federated", version_base=None)
def main(cfg: DictConfig):
    # Parse arguments for experimental settings
    print(OmegaConf.to_yaml(cfg))

    BST_PARAMS = {
    "objective": cfg.bst_params.objective, 
    "num_class": cfg.bst_params.num_class,
    "eta": cfg.bst_params.eta,  # Learning rate
    "max_depth": cfg.bst_params.max_depth,
    "eval_metric": cfg.bst_params.eval_metric,
    "nthread": cfg.bst_params.nthread,
    "num_parallel_tree": cfg.bst_params.num_parallel_tree,
    "subsample": cfg.bst_params.subsample,
    "tree_method": cfg.bst_params.tree_method,
}

    # csv = '/home/pavel/olina/research/xgboost_fl/part-00009-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000 copy.csv'
    csv = '/home/pavel/olina/research/xgboost_fl/benighn_0.csv'
    train_data_list, valid_data_list, test_data = prepare_dataset( csv, 
                                                                    classification_type=cfg.classification_type,#'group', #'binary',
                                                                    mode=cfg.mode_data, #'STRATIFIED' 'SINGLE_ATTACK' 'RANDOM' 'EXCLUDE_SINGLE_ATTACK' 'HALF_BENIGN'
                                                                    type_normalization=cfg.normalization, #'StandardScaler',
                                                                    type_feature_extractor=cfg.feature_extractor,
                                                                    num_partitions=cfg.num_clients,
                                                                    max_partitions_size=cfg.max_partitions_size,
                                                                    val_ratio=cfg.val_ratio
                                                                    )
   
    strategy = instantiate(cfg.strategy,
                        evaluate_fn=get_evaluate_fn(cfg.bst_params, test_data), # централизованная оценка
                        on_evaluate_config_fn=eval_config,
                        on_fit_config_fn=fit_config,
                        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation # федеративная оценка
                        )


    params = BST_PARAMS
    params.update({"eta": cfg.bst_params.eta / cfg.num_clients})

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(
            train_data_list,
            valid_data_list,
            params,
            cfg.num_local_round,
        ),
        num_clients=cfg.num_clients,
         ray_init_args = {
            "include_dashboard": True, # we need this one for tracking
            # "num_cpus": num_cpus,
            # "num_gpus": num_gpus,
            # "memory": ram_memory,
    },
        client_resources={
        "num_cpus": cfg.num_cpus_per_client,
        "num_gpus": 0.125,
    },
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_manager=None,
        
    )

    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pk'
    results = {'history': history, }

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
