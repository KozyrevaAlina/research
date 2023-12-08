from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from federated_learning.model import BiLSTM, train, test # для каждой модели свое будет


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, 
                 ml_model, 
                 trainloader, 
                 vallodaer, 
                 num_classes=8, 
                 input_size=46) -> None: #################
        """
        ml_model - name of model
        num_classes - count of classes for classification: 34, 8, 2
        input_size - count of features, default=46
        """
        super().__init__()

        # the dataloaders that point to the data associated to this client
        # загрузчики данных, которые указывают на данные, связанные с этим клиентом
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        # модель, которая сначала инициализируется случайным образом
        self.model = ml_model(num_classes, input_size) ###############

        # figure out if this client has access to GPU support or not
        # выяснить, есть ли у этого клиента доступ к поддержке графического процессора или нет
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        """Получите параметры и примените их к локальной модели."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        """Извлеките параметры модели и верните их в виде списка массивов."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        """
        Обучайте модель, полученную от сервера (параметры), используя данные.
        принадлежит этому клиенту. После этого отправьте его обратно на сервер
        """
        # copy parameters sent by the server into client's local model
        # копируем параметры, отправленные сервером, в локальную модель клиента
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.

        # извлекаем элементы конфигурации, отправленной сервером. 
        # Обратите внимание, что конфигурация, отправляемая сервером каждый раз, 
        # когда клиенту необходимо участвовать, — это простой, но мощный механизм для 
        # настройки этих гиперпараметров во время процесса FL. 
        # Например, возможно, вы хотите, чтобы клиенты снизили свой LR после нескольких раундов FL.
        # или вы хотите, чтобы клиенты выполняли больше локальных эпох на более поздних этапах моделирования, 
        # вы можете контролировать это, настроив то, что вы передаете в `on_fit_config_fn` 
        # при определении вашей стратегии.

        lr = config["lr"]
        # momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        # очень стандартный оптимизатор
        # optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        input_size = 46

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)

        # провести местное обучение. Эта функция идентична той, которую вы могли использовать 
        # раньше в проектах, не связанных с FL. Для более продвинутой реализации FL вы, 
        # возможно, захотите настроить его, но в целом с точки зрения клиента «локальное обучение» 
        # можно рассматривать как форму «централизованного обучения» с учетом предварительно обученной модели 
        # (т. е. модели, полученной с сервера).

        train(self.model, self.trainloader, optim, epochs, input_size, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)

        # Клиенты Flower должны вернуть три аргумента: 
        # - обновленную модель, 
        # - количество примеров в клиенте (хотя это немного зависит от выбора вами стратегии агрегации) и 
        # - словарь метрик (здесь вы можете добавить любые дополнительные данные, 
        # но эти в идеале являются небольшими структурами данных)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy, precision, recall, f1 = test(self.model, self.valloader, self.input_size, self.device) # задается в модели 
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        return float(loss), len(self.valloader), metrics

def generate_client_fn(ml_model, trainloaders, valloaders, num_classes): ####
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """
    """
    Верните функцию, которая может использоваться VirtualClientEngine.
    для создания FlowerClient с идентификатором клиента `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.

        # Эта функция будет вызываться внутри VirtualClientEngine
        # Каждый раз, когда cid-му клиенту предлагается принять участие в 
        # моделировании FL (будь то выполнение функции fit() или оценка())

         # Возвращает обычный FLowerClient, который будет использовать
         # загрузчики данных cid-th train/val в качестве локальных данных.

        return FlowerClient(
            ml_model, ######
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
        )

    # return the function to spawn client
    # возвращаем функцию для создания клиента
    return client_fn
