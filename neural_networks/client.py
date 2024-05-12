from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import sys
import torch
import flwr as fl

from hydra.utils import instantiate

from model import train, test

from typing import List, Tuple
from flwr.common import Metrics

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client"""

    def __init__(self, 
                 trainloader, 
                 vallodaer, 
                 model_cfg) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = instantiate(model_cfg)

        # self.model_name = model_cfg.get("name", type(self.model).__name__)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        # print('####size_in_bytes',sys.getsizeof(parameters))      

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        # optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        train(self.model, self.trainloader, optim, epochs, self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy, precision, recall, f1 = test(self.model, self.valloader, self.device)
        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return float(loss), len(self.valloader), {  "accuracy": accuracy, 
                                                    "precision": precision,
                                                    "recall": recall,
                                                    "f1": f1}
    

def generate_client_fn(trainloaders, valloaders, model_cfg):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            model_cfg=model_cfg,
        ).to_client()

    # return the function to spawn client
    return client_fn


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
# # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Initialize variables for weighted metrics
    weighted_accuracy = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_examples = 0

    for num_examples, m in metrics:
        total_examples += num_examples

        # Calculate weighted metrics for each client
        weighted_accuracy += num_examples * m["accuracy"]
        weighted_precision += num_examples * m["precision"]
        weighted_recall += num_examples * m["recall"]
        weighted_f1 += num_examples * m["f1"]

    # Calculate overall weighted metrics
    if total_examples > 0:
        weighted_accuracy /= total_examples
        weighted_precision /= total_examples
        weighted_recall /= total_examples
        weighted_f1 /= total_examples

    # Aggregate and return custom metric (weighted averages)
    return {
        "accuracy": round(weighted_accuracy, 4),
        "precision": round(weighted_precision, 4),
        "recall": round(weighted_recall, 4),
        "f1": round(weighted_f1, 4)
    }