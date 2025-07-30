import torch.nn as nn
from einops import rearrange  # Used for flattening feature maps before fully connected layers

# Upgraded CNN model for CIFAR-10
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # First convolutional layer: 3 input channels (RGB), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)      # Batch normalization after conv1 to stabilize training
        self.relu1 = nn.ReLU()             # ReLU activation function (modular version)

        # Second convolutional layer: 32 input channels -> 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)      # Batch normalization again
        self.relu2 = nn.ReLU()

        # Third convolutional layer: 64 input -> 128 output
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)     # BatchNorm on deeper layer
        self.relu3 = nn.ReLU()

        # Fourth convolutional layer: 128 -> 256 output channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        # Max pooling: halves the feature map size each time (stride=2)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout: randomly zeros 50% of the neurons in FC layers (helps generalization)
        self.dropout = nn.Dropout(0.5)

        # After 4 conv+pool blocks: 256 channels, 2x2 feature map → 256*2*2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)         # Second FC layer
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(128, 10)          # Final output layer: 10 classes in CIFAR-10

    def forward(self, x):
        # Conv Block 1
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> Pool

        # Conv Block 2
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> Pool

        # Conv Block 3
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))

        # Conv Block 4
        x = self.pool(self.relu4(self.bn4(self.conv4(x))))

        # Flatten the feature map: (B, C, H, W) -> (B, C*H*W)
        x = rearrange(x, 'b c h w -> b (c h w)')

        # Fully Connected Block with Dropout
        x = self.dropout(self.relu4(self.fc1(x)))  # FC1 + ReLU + Dropout
        x = self.dropout(self.relu5(self.fc2(x)))  # FC2 + ReLU + Dropout

        # Final output (logits for 10 classes)
        x = self.fc3(x)

        return x
