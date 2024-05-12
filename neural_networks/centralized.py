import torch
import pickle
from pathlib import Path

import hydra 
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset
from model import train, test
from monitor import monitor

@hydra.main(config_path="conf", config_name="base_centralized", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    monitor()

    ## 2. Prepare your dataset
    csv = '/home/pavel/olina/research/neural_network_fl/benighn_0.csv'
    trtainloaders, validationloaders, testloader = prepare_dataset( csv, 
                                                                    mode='CENTRALIZED',#cfg.mode_data, #'STRATIFIED' 'SINGLE_ATTACK' 'RANDOM' 'EXCLUDE_SINGLE_ATTACK' 'HALF_BENIGN'
                                                                    type_normalization=cfg.type_normalization, #'StandardScaler',
                                                                    type_feature_extractor=cfg.type_feature_extractor,
                                                                    num_partitions=1,#cfg.num_clients,
                                                                    max_partitions_size=cfg.max_partitions_size,
                                                                    batch_size=cfg.batch_size,
                                                                    val_ratio=cfg.val_ratio
                                                                    )
    # print(len(trtainloaders))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device =  "cpu"

    
    ## 3. Define model
    model_config = cfg.model
    model = hydra.utils.instantiate(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.config_fit.lr, betas=(0.9, 0.999))

    torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)

    ## 4. Train
    history = []
    res, model = train(model, trtainloaders, optimizer, epochs=cfg.num_rounds, device=device)
    history.append('train')
    history.append(res)

    ## 5. Evaluate
    loss, accuracy, precision, recall, f1 = test(model, testloader, device=device)
    history.append('test')
    history.append({"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    print(f" loss: {loss}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
    
    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pk'
    results = {'history': history,}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    main()