import torch
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module): 
    def __init__(self, 
                 num_classes: int=8): # input_size = 46 number of features
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=46,
            hidden_size=32,
            bidirectional=True,
        )
       
        self.fc = nn.Linear(32 * 2, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1) # (sequence_length, batch_size, input_size)
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out[-1]  
        model = self.fc(r_out) 
        return model
    
###########################    
class MLP(nn.Module):
    def __init__(self, input_size=46, hidden_size=32, num_classes=8):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.view(-1, 46)  # Reshape input (batch_size, sequence_length, input_size) to (batch_size, input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#############################

    
def train(net, trainloader, optimizer, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for _ in range(epochs):
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
    return net


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    predicts = []
    all_labels = []

    net.eval()
    net.to(device)
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            features = features.view(-1, 1, 46).float()
            labels = labels.long() 
            outputs = net(features)
            outputs = outputs.float() 
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            #####
            # for metrics   
            # Collect predictions and true labels for later use in calculating metrics
            predicts += predicted.tolist()
            all_labels += labels.cpu().numpy().tolist()
    accuracy = correct / len(testloader.dataset)
    # accuracy = accuracy_score(all_labels, predicts)
    # precision = precision_score(all_labels, predicts, average='weighted', zero_division=1)
    # recall = recall_score(all_labels, predicts, average='weighted', zero_division=1)
    # f1 = f1_score(all_labels, predicts, average='weighted')
    
    return loss, accuracy#, precision, recall, f1
    



