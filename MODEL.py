import torch
import torch.nn as nn
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(6272, 3000)    # 7 * 7 * 128 = 6272
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.conv1(x) # 28 * 28 * 32
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 14 * 14 * 32
        x = self.conv2(x) # 14 * 14 * 64
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 7 * 7 * 64
        x = self.dropout(x)
        x = self.conv3(x) # 7 * 7 * 128
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # = 6272
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
