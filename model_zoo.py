import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        
        self.fc1 = nn.Linear(64 *3 *3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.max_pool2d(x, 2)     
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, 2)     
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x))  
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.shape[0],-1)  
        x = F.relu(self.fc1(x))     
        x = F.relu(self.fc2(x))     
        x = self.fc3(x)             
        return x
    
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, transform=None):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.transform = transform

    def forward(self, x):
        if self.transform:
            x = self.transform(x)
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.hidden1(x))
        x = self.output(x)
        x = F.log_softmax(x,dim = 1)
        return x