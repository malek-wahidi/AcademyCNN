
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256) 
        
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 256 * 4 * 4)
        self.fc2 = nn.Linear(256 * 4 * 4, 128 * 4 * 4)
        self.fc3 = nn.Linear(128 * 4 * 4, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.bn1(self.conv4(x)))
        x = F.relu(self.bn1(self.conv5(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv6(x)))
        x = F.relu(self.bn2(self.conv7(x)))
        x = self.pool(x)

        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
