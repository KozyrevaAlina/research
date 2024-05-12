from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
from flwr.common.parameter import ndarrays_to_parameters

class LSTM(nn.Module):
    def __init__(self, 
                 input_size=46, 
                 hidden_size=64, 
                 num_classes=8):
        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output, _ = self.lstm(x) # Input shape: (batch_size, sequence_length, input_size)
        output = output[:, -1, :] # Take the last output from LSTM as the representation of the sequence
        output = self.fc(output) # Fully connected layer
        output = self.softmax(output) # Softmax activation for classification
        return output
    
class BiLSTM(nn.Module): 
    def __init__(self, 
                 input_size= 46, # number of features
                 hidden_size=64, ## 
                 num_classes: int=8): 
        
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True, # 
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # self.dropout = nn.Dropout(0.2) ####
        # self.l1_strength = 0.0001
        # self.l2_strength = 0.0001

    def forward(self, x):
        x = x.transpose(0, 1) # (sequence_length, batch_size, input_size)
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out[-1]  
        model = self.fc(r_out) 
###############
        # model = self.dropout(model) ####
        # # Рассчитываем L1 и L2 регуляризацию
        # l1_regularization = torch.tensor(0., requires_grad=True)
        # l2_regularization = torch.tensor(0., requires_grad=True)
        # for param in self.parameters():
        #     l1_regularization += torch.norm(param, 1)
        #     l2_regularization += torch.norm(param, 2)
        
        # # Добавляем регуляризацию к потерям
        # model += self.l1_strength * l1_regularization
        # model += 0.5 * self.l2_strength * l2_regularization
###############
        probs = F.softmax(model, dim=1)  # применить softmax к выходам модели #######
        # return model
        return probs

class GRU(nn.Module):
    def __init__(self, input_size=46, hidden_size=64, num_classes=8):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        output, _ = self.gru(x)
        # Take the last output from GRU as the representation of the sequence
        output = output[:, -1, :]
        # Fully connected layer
        output = self.fc(output)
        # Softmax activation for classification
        output = self.softmax(output)
        return output

class MLP(nn.Module):
    def __init__(self, 
                 input_size=46, 
                 hidden_size=64, 
                 num_classes=8):
        
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1) # new
        return x

class CNN(nn.Module):
    def __init__(self, 
                 input_size=46, 
                 num_classes=8):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1) # new
        return x


class EnsembleModel(nn.Module):   
    def __init__(self, 
                 model1=LSTM(), 
                 model2=MLP(), 
                 model3=CNN(),
                 num_classes=8,
                 ):
        
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.classifier = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out          
    

def train(net, trainloader, optimizer, epochs,  device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss() 
    net.train()
    net.to(device)

    predicts = []
    all_labels = []
    train_result = []

    train_f1 = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            features = features.view(-1, 1, 46).float() # for transformation

            labels = labels.long() ######
            optimizer.zero_grad()
            outputs = net(features)
            outputs = outputs.float() ######
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #####
            # for metrics   
            # Collect predictions and true labels for later use in calculating metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            predicts += predicted.tolist()
            all_labels += labels.cpu().numpy().tolist()

        train_f1 = round(f1_score(all_labels, predicts, average='weighted'), 4)
        train_accuracy = round(accuracy_score(all_labels, predicts), 4)
        train_precision = round(precision_score(all_labels, predicts, average='weighted', zero_division=1), 4)
        train_recall = round(recall_score(all_labels, predicts, average='weighted', zero_division=1), 4)
        res = {"epoch": epoch+1, "loss": round(train_loss, 4), "train_accuracy": train_accuracy, "train_precision": train_precision, "train_recall": train_recall, "f1": train_f1}
        train_result.append(res)
    
    return train_result, net  

def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0

    predicts = []
    all_labels = []

    net.eval()
    net.to(device)
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            features = features.view(-1, 1, 46).float() #

            labels = labels.long() 
            outputs = net(features)
            outputs = outputs.float() 
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == labels).sum().item()

            #####
            # for metrics   
            # Collect predictions and true labels for later use in calculating metrics
            predicts += predicted.tolist()
            all_labels += labels.cpu().numpy().tolist()
    accuracy = round(accuracy_score(all_labels, predicts), 4)
    precision = round(precision_score(all_labels, predicts, average='weighted', zero_division=1), 4)
    recall = round(recall_score(all_labels, predicts, average='weighted', zero_division=1), 4)
    f1 = round(f1_score(all_labels, predicts, average='weighted'), 4)
    
    return round(loss, 4), accuracy, precision, recall, f1
    


def model_to_parameters(model):
    """Note that the model is already instantiated when passing it here.

    This happens because we call this utility function when instantiating the parent
    object (i.e. the FedAdam strategy in this example).
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters
