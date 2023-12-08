# normalisation
# feature extracture
# + func from dataset fl

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis, IncrementalPCA, FastICA, PCA
# from Kpca import Kpca #???
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils import resample

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, index):  
        return self.features[index], self.labels[index]

#### Сделать маппинг лэйблов


# mode of creating datasets for federated learning
MODE = ['STRATIFIED', 'SINGLE_ATTACK', 'EXCLUDE_SINGLE_ATTACK', 'HALF_BENIGN', 'RANDOM']
RANDOM_STATE = 42

def prepare_dataset(
    df: pd.DataFrame,
    mode: str = None, # 'STRATIFIED',
    normalization: str = 'StandardScaler',
    feature_extractor: str = None, #'IncrementalPCA',
    num_features: int = 46,
    num_partitions: int = 10,
    max_partitions_size: int = 300, # (train + val)
    batch_size: int = 64, 
    val_ratio: float = 0.1
):
    """
    df: full dataset for train and test
    mode: mode of creating datasets for federated learning
          MODE = ['STRATIFIED', 'SINGLE_ATTACK', 'EXCLUDE_SINGLE_ATTACK', 'HALF_BENIGN', 'RANDOM']
    normalization: type of data normalization
    num_partitions: number of client in federated learning (one per client)
    max_partitions_size: max size of dataset for one client
    batch_size:
    val_ratio: size of validation datasey
    """
    X = df.iloc[:, 0:-1].values # [:, 0:-1]: выбирает все строки (:) и все столбцы, кроме последнего (0:-1)
    y = df['label'].values

    # create trainset and testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # add normalization
    # if normalization == 'Standard':
        # scaler = StandardScaler()
    # elif normalization == 'MinMax':
        # scaler = MinMaxScaler()
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
 
    # add feature extractor
    if feature_extractor == 'IncrementalPCA':
        ipca = IncrementalPCA(n_components=num_features)
        # ipca.partial_fit(X_train)
        X_train = ipca.fit_transform(X_train)
        X_test = ipca.transform(X_test)

    elif feature_extractor == 'PCA':
        pca = PCA(n_components=num_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # elif feature_extractor == 'LinearDiscriminantAnalysis':
    #     lda = LinearDiscriminantAnalysis(n_components=num_features)
    #     X_train = lda.fit_transform(X_train, y_train)
    #     X_test = lda.transform(X_test)


    if mode == 'STRATIFIED':
        """
        stratified splint full dataset into num_partitions
        """
        trainloaders = []
        valloaders = []

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_part, y_train_part = X_train[train_index], y_train[train_index]
            trainset_for_one = np.column_stack([X_train_part, y_train_part])
            
            max_len_val = int(len(trainset_for_one) * val_ratio)
            if len(trainset_for_one) > (max_partitions_size - max_len_val):
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=RANDOM_STATE)

            trainset_for_one = CustomDataset(trainset_for_one[:, :-1], trainset_for_one[:, -1])
            valset_for_one = CustomDataset(valset_for_one[:, :-1], valset_for_one[:, -1])
            
            trainloader = DataLoader(trainset_for_one, batch_size=64, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=64, shuffle=True)
            
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)
       
    elif mode == 'SINGLE_ATTACK':
        # for case when num_partitions (num_clients) > num_attacks
        trainset = np.column_stack([X_train, y_train])

        # определим нужные размерности для деления наборов данных
        num_attacks = len(np.unique(trainset[:, -1]))
        # сколько раз нацело разделить каждый набор атак
        size_split = num_partitions // num_attacks

        # сколько из разделенных ранее наборов еще нужно разделить
        remainder = num_partitions - (size_split * num_attacks)

        sets_single_attacks = []

        # разделим общий датасет на датасеты атак
        for label in np.unique(trainset[:, -1]): # y
            set_attack = trainset[trainset[:, -1] == label]
            sets_single_attacks.append(set_attack)       

        sorted_sets_single_attacks = sorted(sets_single_attacks, key=len, reverse=True)
        sets_out = []
        for i, subset in enumerate(sorted_sets_single_attacks):
            if i < remainder:
                # Разделите датасет попалам
                half = int(len(subset)/2)
                # запишем в финальный сет
                sets_out.append(subset[:-half])
                sets_out.append(subset[-half:])
            else:
                # допишем в финальный сет оставшиеся наборы атак
                sets_out.append(subset)

        #  Создайте DataLoader для каждого тренировочного  и валидационного набора
        trainloaders = []
        valloaders = []

        for subset in sets_out:
            if len(subset) > max_partitions_size:
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)

            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_part)
            
            # train и val для одного клиента
            trainset_for_one = CustomDataset(X_train, y_train)
            valset_for_one = CustomDataset(X_val, y_val)

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)

    elif mode == 'EXCLUDE_SINGLE_ATTACK':
        # for case when num_partitions (num_clients) > num_attacks
        trainset = np.column_stack([X_train, y_train])
        # определим нужные размерности для деления наборов данных
        num_attacks = len(np.unique(trainset[:, -1]))

        # сколько раз нацело разделить каждый набор атак
        size_split = num_partitions // num_attacks

        # сколько из разделенных ранее наборов еще нужно разделить
        remainder = num_partitions - (size_split * num_attacks)

        sets_exclude_single_attacks = []
        # разделим общий датасет на датасеты атак
        for label in np.unique(trainset[:, -1]): # y
            set_exclude_attack = trainset[trainset[:, -1] != label] 
            sets_exclude_single_attacks.append(set_exclude_attack)       

        sorted_sets_exclude_single_attacks = sorted(sets_exclude_single_attacks, key=len, reverse=True)
        sets_out = []
        for i, subset in enumerate(sorted_sets_exclude_single_attacks):
            if i < remainder:
                # Разделите датасет попалам
                half = int(len(subset)/2)
                # запишем в финальный сет
                sets_out.append(subset[:-half])
                sets_out.append(subset[-half:])
            else:
                # допишем в финальный сет оставшиеся наборы атак
                sets_out.append(subset)

        # Создайте DataLoader для каждого тренировочного  и валидационного набора
        trainloaders = []
        valloaders = []

        for subset in sets_out:
            if len(subset) > max_partitions_size:
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)
            
            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=RANDOM_STATE) #, stratify=y_part)
            
            trainset_for_one = CustomDataset(X_train, y_train)
            valset_for_one = CustomDataset(X_val, y_val)

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader) 

    elif mode == 'HALF_BENIGN':
        # на сколько частей делим весь трафик и BENIGN
        if num_partitions % 2 == 0:
            size_split_all = num_partitions // 2
            size_split_benign = size_split_all
        else:
            size_split_all = num_partitions // 2
            size_split_benign = num_partitions - size_split_all

        trainloaders = []
        valloaders = []

        # benign trainset
        trainset = np.column_stack([X_train, y_train])
        trainset_benign = trainset[trainset[:, -1] == 0] ###### ИСПРАВИТЬ
        X_benign = trainset_benign[:, :-1]
        y_benign = trainset_benign[:, -1]

        skf = StratifiedKFold(n_splits=size_split_benign, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(X_benign, y_benign):
            X_train_part, y_train_part = X_train[train_index], y_train[train_index]
            trainset_for_one = np.column_stack([X_train_part, y_train_part])

            max_len_val = int(len(trainset_for_one) * val_ratio)
            if len(trainset_for_one) > (max_partitions_size - max_len_val):
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=RANDOM_STATE)

            trainset_for_one = CustomDataset(trainset_for_one[:, :-1], trainset_for_one[:, -1])
            valset_for_one = CustomDataset(valset_for_one[:, :-1], valset_for_one[:, -1])

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)   

        # full trainset
        skf = StratifiedKFold(n_splits=size_split_all, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X_train, y_train):
            X_train_part, y_train_part = X_train[train_index], y_train[train_index]
            trainset_for_one = np.column_stack([X_train_part, y_train_part])
            
            max_len_val = int(len(trainset_for_one) * val_ratio)
            if len(trainset_for_one) > (max_partitions_size - max_len_val):
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=RANDOM_STATE)

            trainset_for_one = CustomDataset(trainset_for_one[:, :-1], trainset_for_one[:, -1])
            valset_for_one = CustomDataset(valset_for_one[:, :-1], valset_for_one[:, -1])

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)

    elif mode == 'RANDOM':
        trainset = np.column_stack([X_train, y_train])

        # Разделите датасет на n частей
        # Получите индексы датафрейма
        indices = np.arange(len(trainset))

        # Перемешайте индексы случайным образом
        shuffled_indices = np.random.permutation(indices)

        # Создайте перемешанный trainset
        shuffled_trainset = trainset[shuffled_indices]

        # Разделите индексы на n частей, чтобы они не пересекались
        trainsets = np.array_split(shuffled_indices, num_partitions)

        trainloaders = []
        valloaders = [] 

        calculated_size = len(trainset) // num_partitions

        for dataset_indices in trainsets:
            # Выбор соответствующих строк из датафрейма df
            subset = shuffled_trainset[dataset_indices]

            # Ограничиваем размер нового датасета заданной длиной класса
            # если длина датасета при его делении на num_partitions частей больше, чем max_partitions_size
            if calculated_size > max_partitions_size:
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=RANDOM_STATE)

            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=RANDOM_STATE) #, stratify=y_part)

            trainset_for_one = CustomDataset(X_train, y_train)
            valset_for_one = CustomDataset(X_val, y_val)

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)

    else:
        # trainloaders = []
        # valloaders = []

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_train)
        
        trainset = CustomDataset(X_train, y_train)
        valset = CustomDataset(X_val, y_val)
        
        trainloaders = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloaders = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # trainloaders.append(trainloader)
        # valloaders.append(valloader)
        # print('You do not choose dataset creation mode or choose it wrong.')
 
    # leave the test set intact 
    testset = CustomDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader



