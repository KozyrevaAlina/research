# from sklearn.base import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
import torch
import pickle
from pathlib import Path
import numpy as np

import hydra 
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import xgboost as xgb

from dataset import prepare_dataset
from monitor import monitor


@hydra.main(config_path="conf", config_name="base_centralized", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    monitor()

    ## 2. Prepare your dataset
    # csv = '/home/pavel/olina/research/neural_network_fl/part-00009-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
    csv = '/home/pavel/olina/research/xgboost_fl/benighn_0.csv'
    trainsets, vallsets, testset = prepare_dataset( csv, 
                                                                    mode='CENTRALIZED',#cfg.mode_data, #'STRATIFIED' 'SINGLE_ATTACK' 'RANDOM' 'EXCLUDE_SINGLE_ATTACK' 'HALF_BENIGN'
                                                                    type_normalization=cfg.type_normalization, #'StandardScaler',
                                                                    type_feature_extractor=cfg.type_feature_extractor,
                                                                    num_partitions=1,#cfg.num_clients,
                                                                    max_partitions_size=cfg.max_partitions_size,
                                                                    batch_size=cfg.batch_size,
                                                                    val_ratio=cfg.val_ratio
                                                                    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    ## 3. Define model
    # Определение параметров модели XGBoost
    params = {
        'objective': 'multi:softmax',  # Многоклассовая классификация
        'num_class': 8,  # Количество классов
        'eta': 0.1,  # Скорость обучения (learning rate)
        'max_depth': 8,  # Максимальная глубина дерева
        'eval_metric': 'auc',  # Метрика оценки качества модели
        'nthread': 16,  # Количество потоков для параллельной работы
        'num_parallel_tree': 1,  # Количество параллельных деревьев
        'subsample': 1,  # Доля случайной выборки обучающих данных
        'tree_method': 'hist'  # Метод построения деревьев
    }

    ## 4. Train
    history = []
    # Преобразование данных в формат DMatrix для XGBoost
    # Извлечение массива из списка trainsets
    train_data = trainsets[0]

    # Разделение признаков и меток
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # Преобразование в объекты DMatrix
    train_dmatrix = xgb.DMatrix(x_train, label=y_train)

    # шаги для тестовых данных
    test_data = testset
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    test_dmatrix = xgb.DMatrix(x_test, label=y_test)

    # Обучение модели XGBoost
    num_round = 10  # Количество итераций обучения
    bst = xgb.train(params, train_dmatrix, num_round)
    # print(bst)
    history.append('train')
    # history.append(res)

    ## 5. Evaluate
    # Предсказание на тестовом наборе данных
    # Получить список всех классов из истинных меток
    y_pred = bst.predict(test_dmatrix)
    
    # Вычисление метрик
    accuracy = accuracy_score(test_dmatrix.get_label(), y_pred)
    precision = precision_score(test_dmatrix.get_label(), y_pred, average='weighted')
    recall = recall_score(test_dmatrix.get_label(), y_pred, average='weighted')
    f1 = f1_score(test_dmatrix.get_label(), y_pred, average='weighted')

    # loss = log_loss(test_dmatrix.get_label(), y_pred, labels=classes) #test_dmatrix.get_label()
    history.append('test')
    history.append({ "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
    
    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pk'
    results = {'history': history,}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()