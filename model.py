import torch.nn as nn #torch.nn contains conv2d,linear...
import torch.nn.functional as F #for relu

class BetterNet(nn.Module):
    def __init__(self): #constructor
        super(BetterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) #3 for RGB (input),64 feature maps(output),padding=1 for keeping the same size
        self.bn1 = nn.BatchNorm2d(64) #batch normalization for 64 channels (scaling data between -1,1)


        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2) #reduce the spatial size by 2

        self.dropout1 = nn.Dropout(0.3) #randomly dropout 30% of the neurons to prevent overfitting
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10) #10:one for each cifar-10 class

    def forward(self, x): #defines how data flows through the layers
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #convolution->ReLU->BatchNorm->Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 256 * 4 * 4) #flatten for fully connected layers

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        #Apply dropout → ReLU → dropout again → output logits.

        return x
        #Output is a vector of 10 scores (one per class).