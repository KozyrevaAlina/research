import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Note the model and functions here defined do not have any FL-specific components.
# Обратите внимание, что определенные здесь модель и функции не содержат каких-либо компонентов, специфичных для FL.

class CNN(nn.Module):
    """A simple CNN suitable for simple vision tasks."""
    """Простой CNN, подходящий для простых задач машинного зрения."""

    def __init__(self, num_classes: int) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BiLSTM(nn.Module): 
    def __init__(self, 
                 num_classes: int=8, 
                 input_size: int=46): # input_size = 46 number of features
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            bidirectional=True,
        )
       
        self.fc = nn.Linear(32 * 2, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1) # (sequence_length, batch_size, input_size)
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out[-1]  
        r_out = self.fc(r_out) 
        return r_out


def train(model, 
          trainloader, 
          optimizer, 
          epochs: int, 
          input_size: int, 
          device: str,
          verbose=False):
    """Train the network on the training set.
    This is a fairly simple training loop for PyTorch.
    """

    """Обучаем сеть на обучающем наборе.

    Это довольно простой цикл обучения для PyTorch.
    """
    # predicts = []
    # labels = []

    # input_size = 46 # number of features
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for data in trainloader:
            features, labels = data
            # labels = torch.tensor([int(label) for label in labels])

            features = features.view(-1, 1, input_size).float() # for transformation
            # features = features.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # metrics
            # epoch_loss += loss
            # _, predicted = torch.max(outputs.data, 1)

            # for metrics   
            # Collect predictions and true labels for later use in calculating metrics
            # predicts.extend(predicted)
            # labels.extend(y.numpy())

        # accuracy = accuracy_score(labels, predicts)
        # precision = precision_score(labels, predicts, average='weighted')
        # recall = recall_score(labels, predicts, average='weighted')
        # f1 = f1_score(labels, predicts, average='weighted')

        # if verbose:
            # print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {accuracy}, precision {precision}, recall {recall}, f1 {f1}")


def test(model, 
         testloader,  
         input_size: int,  
         device: str):
    """Validate the network on the entire test set.
    and report loss and metrics.
    """
    """Проверьте сеть на всем тестовом наборе.

    и сообщать о потерях и точности.
    """
    # input_size = 46 # number of features
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    predicts = []
    labels = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            features, labels = data
            # labels = torch.tensor([int(label) for label in labels])
            features = features.view(-1, 1, input_size).float() # for transformation
            # features = features.to(device)

            outputs = model(features)

            loss += criterion(outputs, labels).item()
            # _, predicted = torch.max(outputs.data, 1)

            # for metrics   
            # Collect predictions and true labels for later use in calculating metrics
            # predicts.extend(predicted)
            # labels.extend(y.numpy())

    # accuracy = accuracy_score(labels, predicts)
    # precision = precision_score(labels, predicts, average='weighted')
    # recall = recall_score(labels, predicts, average='weighted')
    # f1 = f1_score(labels, predicts, average='weighted')

    return loss #, accuracy, precision, recall, f1
