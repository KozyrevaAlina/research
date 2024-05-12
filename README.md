# research
# Обнаружение вторжений в децентрализованных информационных системах на основе методов федеративного машинного обучения

## ЦЕЛЬ:
Разработка СОВ для децентрализованных информационных систем на примере интернета вещей (IoT).

## Исследуемый набор данных:
[CiCIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

## Используемый framework федеративного обучения:
[flower.dev](https://flower.dev)

## data_exploratin – исследование данных
- `EDA_feature_importance.ipynb`

## xgboost – архитектура федеративного и централизованного обучения для модели xgboost
- `centralized.py` - архитектура централизованного обучения
- `federated.py` - архитектура федеративного обучения
- `dataset.py` - предобработка данных
- `client.py` - клиент федеративного обучения
- `server.py` - сервер федеративного обучения
- `utils.py` - вспомогательные функции
- `monitor.py` - функции отслеживания задействования вычислительных ресурсов
- `conf`: - задание конфигурации моделеи и процесса обучения
  - `strategy`:
    - `fedxgbbagging.yaml`   
  - `base_centralized.yaml`
  - `base_federated.yaml` 
  

## neural_networks – архитектура федеративного и централизованного обучения для нейронных моделей

## Experiments – результаты экспериментов
- `Classic ML_8.ipynb`
- `Classic ML_8_part_data.ipynb`
- `Classic ML_8_part_data_single.ipynb`
- `FL_8_EXCLUDE_SINGLE_ATTACK.ipynb`
- `FL_8_EXCLUDE_SINGLE_ATTACK_part.ipynb`
- `FL_8_HALF_BENIGN.ipynb`
- `FL_8_HALF_BENIGN_part.ipynb`
- `FL_8_SINGLE_ATTACK.ipynb`
- `FL_8_SINGLE_ATTACK_part.ipynb`
- `FL_8_STRATIFIED.ipynb`
- `FL_8_STRATIFIED_3clients.ipynb`
- `FL_8_STRATIFIED_5clients.ipynb`
- `FL_8_STRATIFIED_part.ipynb`
- `FL_8_STRATIFIED_part_5clients.ipynb`
