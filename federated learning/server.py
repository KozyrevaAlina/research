from collections import OrderedDict


from omegaconf import DictConfig

import torch

from federated_learning.model import BiLSTM, test


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""
    """Функция возврата, которая подготавливает конфигурацию для отправки клиентам."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        # Эта функция будет выполняться стратегией в ее метод `configure_fit()`.

        # Здесь мы возвращаем одну и ту же конфигурацию в каждом раунде, 
        # но здесь вы можете использовать входной аргумент `server_round`, 
        # чтобы со временем адаптировать эти настройки для клиентов. 
        # Например, вы можете захотеть, чтобы клиенты использовали 
        # другую скорость обучения на более поздних этапах процесса FL 
        # (например, меньший lr после N раундов).
        # return {
        #     "lr": config.lr,
        #     # "momentum": config.momentum,
        #     "local_epochs": config.local_epochs,
        # }
        return {
            "lr": 0.01,
            # "momentum": config.momentum,
            "local_epochs": 1,
        }

        # config = {
        # "epochs": 1,  # Number of local epochs done by clients
        # "lr": 0.01,  # Learning rate to use by clients during fit()
        # }
    return fit_config_fn


def get_evaluate_fn(ml_model, 
                    num_classes: int, 
                    input_size: int,
                    testloader): ############
    """Define function for global evaluation on the server."""
    """Определить функцию для глобальной оценки на сервере."""

    def evaluate_fn( # ml_model, 
                    server_round: int, 
                #    input_size: int,
                    parameters, 
                    config
                    ):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        # Эта функция вызывается методом стратегии `evaluate()` и получает 
        # в качестве входных аргументов текущий раундовый номер и параметры глобальной модели.
        # эта функция принимает эти параметры и оценивает глобальную модель в наборе оценочных/тестовых данных.

        # model = Net(num_classes)
        model = ml_model(num_classes, input_size) ###############

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.

        # Здесь мы оцениваем глобальную модель на тестовом наборе. 
        # Напомним, что в более реалистичных условиях вы будете делать это только 
        # в конце эксперимента с FL и можете использовать входной аргумент `server_round`, 
        # чтобы определить, является ли это последним раундом. 
        # Если это не так, то желательно использовать глобальный набор проверки.
        loss, accuracy, precision, recall, f1 = test(model, testloader, input_size, device)
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.

        # Сообщите о потерях и любых других показателях (внутри словаря). 
        # В этом случае мы сообщаем о глобальной точности теста.
        return loss, metrics
    
    return evaluate_fn
