import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis, IncrementalPCA, FastICA, PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils import resample

from torch.utils.data import DataLoader, Dataset


# convert data types to reduce memory usage
dict_types = {
          'flow_duration': np.float32, 
          'Header_Length': np.int32, 
          'Protocol Type': np.float32, 
          'Duration': np.float32, 
          'Rate': np.uint32, 
          'Srate': np.uint32, 
          'Drate': np.float32, 
          'fin_flag_number': np.uint8, 
          'syn_flag_number': np.uint8, 
          'rst_flag_number': np.uint8, 
          'psh_flag_number': np.uint8, 
          'ack_flag_number': np.uint8, 
          'ece_flag_number': np.uint8, 
          'cwr_flag_number': np.uint8, 
          'ack_count': np.float16, 
          'syn_count': np.float16, 
          'fin_count': np.uint16, 
          'urg_count': np.uint16, 
          'rst_count': np.uint16, 
          'HTTP': np.uint8, 
          'HTTPS': np.uint8, 
          'DNS': np.uint8, 
          'Telnet': np.uint8, 
          'SMTP': np.uint8, 
          'SSH': np.uint8, 
          'IRC': np.uint8, 
          'TCP': np.uint8, 
          'UDP': np.uint8,
          'DHCP': np.uint8, 
          'ARP': np.uint8, 
          'ICMP': np.uint8, 
          'IPv': np.uint8, 
          'LLC': np.uint8, 
          'Tot sum': np.float32, 
          'Min': np.float32, 
          'Max': np.float32, 
          'AVG': np.float32, 
          'Std': np.float32, 
          'Tot size': np.float32, 
          'IAT': np.float32, 
          'Number': np.float32, 
          'Magnitue': np.float32, 
          'Radius': np.float32, 
          'Covariance': np.float32, 
          'Variance': np.float32, 
          'Weight': np.float32,
          'label': np.uint8
          }

def convert_type(
    df: pd.DataFrame
    ) -> pd.DataFrame: 
    """
    convert data type yo reduce memory usage
    """
    # convert type
    for col, type in dict_types.items():
        df[col] = df[col].astype(type)

    # format column
    # df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

# Creating a dictionary of attack types for 33 attack classes + 1 for benign traffic
dict_34_classes = {'BenignTraffic': 0 ,                                                                                                                         # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':2,  'DDoS-SYN_Flood':3, 'DDoS-UDP_Flood':4, 'DDoS-TCP_Flood':5, 
                   'DDoS-ICMP_Flood':6, 'DDoS-SynonymousIP_Flood':7, 'DDoS-ACK_Fragmentation':8, 'DDoS-UDP_Fragmentation':9, 'DDoS-ICMP_Fragmentation':10, 
                   'DDoS-SlowLoris':11, 'DDoS-HTTP_Flood':12,                                                                                                   # DDoS
                   'DoS-UDP_Flood':13, 'DoS-SYN_Flood':14, 'DoS-TCP_Flood':15, 'DoS-HTTP_Flood':16,                                                             # DoS
                   'Mirai-greeth_flood': 17, 'Mirai-greip_flood': 18, 'Mirai-udpplain': 19,                                                                     # Mirai 
                   'Recon-PingSweep': 20, 'Recon-OSScan': 21, 'Recon-PortScan': 22, 'VulnerabilityScan': 23, 'Recon-HostDiscovery': 24,                         # Reconnaissance
                   'DNS_Spoofing': 25, 'MITM-ArpSpoofing': 26,                                                                                                  # Spoofing
                   'BrowserHijacking': 27, 'Backdoor_Malware': 28, 'XSS': 29, 'Uploading_Attack': 30, 'SqlInjection': 31, 'CommandInjection': 32,               # Web
                   'DictionaryBruteForce': 33}                                                                                                                  # Brute Force 

dict_8_classes = {'BenignTraffic': 0 ,                                                                                                                          # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':1,  'DDoS-SYN_Flood':1, 'DDoS-UDP_Flood':1, 'DDoS-TCP_Flood':1, 
                   'DDoS-ICMP_Flood':1, 'DDoS-SynonymousIP_Flood':1, 'DDoS-ACK_Fragmentation':1, 'DDoS-UDP_Fragmentation':1, 'DDoS-ICMP_Fragmentation':1, 
                   'DDoS-SlowLoris':1, 'DDoS-HTTP_Flood':1,                                                                                                     # DDoS
                   'DoS-UDP_Flood':2, 'DoS-SYN_Flood':2, 'DoS-TCP_Flood':2, 'DoS-HTTP_Flood':2,                                                                 # DoS
                   'Mirai-greeth_flood': 3, 'Mirai-greip_flood': 3, 'Mirai-udpplain': 3,                                                                        # Mirai 
                   'Recon-PingSweep': 4, 'Recon-OSScan': 4, 'Recon-PortScan': 4, 'VulnerabilityScan': 4, 'Recon-HostDiscovery': 4,                              # Reconnaissance
                   'DNS_Spoofing': 5, 'MITM-ArpSpoofing': 5,                                                                                                    # Spoofing
                   'BrowserHijacking': 6, 'Backdoor_Malware': 6, 'XSS': 6, 'Uploading_Attack': 6, 'SqlInjection': 6, 'CommandInjection': 6,                     # Web
                   'DictionaryBruteForce': 7}                                                                                                                                   # 7 - Brute Force

dict_2_classes = {'BenignTraffic': 0 ,                                                                                                                          # Benign traffic
                   'DDoS-RSTFINFlood' :1, 'DDoS-PSHACK_Flood':1,  'DDoS-SYN_Flood':1, 'DDoS-UDP_Flood':1, 'DDoS-TCP_Flood':1, 
                   'DDoS-ICMP_Flood':1, 'DDoS-SynonymousIP_Flood':1, 'DDoS-ACK_Fragmentation':1, 'DDoS-UDP_Fragmentation':1, 'DDoS-ICMP_Fragmentation':1, 
                   'DDoS-SlowLoris':1, 'DDoS-HTTP_Flood':1,                                                                                                     # DDoS
                   'DoS-UDP_Flood':1, 'DoS-SYN_Flood':1, 'DoS-TCP_Flood':1, 'DoS-HTTP_Flood':1,                                                                 # DoS
                   'Mirai-greeth_flood': 1, 'Mirai-greip_flood': 1, 'Mirai-udpplain': 1,                                                                        # Mirai 
                   'Recon-PingSweep': 1, 'Recon-OSScan': 1, 'Recon-PortScan': 1, 'VulnerabilityScan': 1, 'Recon-HostDiscovery': 1,                              # Reconnaissance
                   'DNS_Spoofing': 1, 'MITM-ArpSpoofing': 1,                                                                                                    # Spoofing
                   'BrowserHijacking': 1, 'Backdoor_Malware': 1, 'XSS': 1, 'Uploading_Attack': 1, 'SqlInjection': 1, 'CommandInjection': 1,                     # Web
                   'DictionaryBruteForce': 1} 

def convert_to_gigital_target(
        df: pd.DataFrame,
        classification_type: str,
)-> pd.DataFrame:
    """
    convert label from object or string to digital

    classification_type: binary, group, individual
    """

    if classification_type == 'binary':
        df['label'] = df['label'].map(dict_2_classes)
    elif classification_type == 'group':
        df['label'] = df['label'].map(dict_8_classes)
    elif classification_type == 'individual':
        df['label'] = df['label'].map(dict_34_classes)
    return df


def load_dataset(csv: str,
                 classification_type='group'):
    """
    
    classification_type: binary, group, individual
    
    """
    
    df = pd.read_csv(csv)
    df = convert_to_gigital_target(df, classification_type=classification_type)
    df = convert_type(df)

    return df


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, index):  
        return self.features[index], self.labels[index]


def prepare_dataset(
    csv: str,
    classification_type='group',
    mode: str = None,
    normalization: str = 'StandardScaler',
    feature_extractor: str = None, #'IncrementalPCA',
    num_features: int = 46,
    num_partitions: int = 10,
    max_partitions_size: int = 250_000, # (train + val)
    batch_size: int = 64, 
    val_ratio: float = 0.1
):
    """
    csv: name of csv file with dataset for train and test
    mode: mode of creating datasets for federated learning
          MODE = ['STRATIFIED', 'SINGLE_ATTACK', 'EXCLUDE_SINGLE_ATTACK', 'HALF_BENIGN', 'RANDOM']
    normalization: type of data normalization
    feature_extractor: type of feature extractor
    num_features: number of features (features in dataset)
    num_partitions: number of client in federated learning (one per client)
    max_partitions_size: max size of dataset for one client
    batch_size:
    val_ratio: size of validation datasey
    """

    df = load_dataset(csv, classification_type=classification_type)

    X = df.iloc[:, 0:-1].values # [:, 0:-1]: выбирает все строки (:) и все столбцы, кроме последнего (0:-1)
    y = df['label'].values

    # create trainset and testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
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
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=42)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=42)

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
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=42)

            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=42, stratify=y_part)
            
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
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=42)
            
            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=42) #, stratify=y_part)
            
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
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=42)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=42)

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
                trainset_for_one = resample(trainset_for_one, replace=False, n_samples=max_partitions_size, random_state=42)
            
            X_val_part, y_val_part = X_train[test_index], y_train[test_index]
            valset_for_one = np.column_stack([X_val_part, y_val_part])
            
            if len(valset_for_one) > max_len_val:
                valset_for_one = resample(valset_for_one, replace=False, n_samples=max_len_val, random_state=42)

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
                subset = resample(subset, replace=False, n_samples=max_partitions_size, random_state=42)

            # разделим set каждого клиента на train и val
            X_part = subset[:, :-1]
            y_part = subset[:, -1]
            
            
            X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=val_ratio, random_state=42) #, stratify=y_part)

            trainset_for_one = CustomDataset(X_train, y_train)
            valset_for_one = CustomDataset(X_val, y_val)

            trainloader = DataLoader(trainset_for_one, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset_for_one, batch_size=batch_size, shuffle=True)
            
            trainloaders.append(trainloader)
            valloaders.append(valloader)

    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train)
        
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



