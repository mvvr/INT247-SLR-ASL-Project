import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 4, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 4, stride=2, padding=2)
        self.conv7 = nn.Conv2d(128, 256, 4, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 4, stride=2, padding=2)

        self.norm = nn.BatchNorm2d(256)
    
        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 29)

        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.norm(x)

        x = x.view(-1, 256*4*4)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x