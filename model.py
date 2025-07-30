
import torch.nn as nn #for layers
import torch.nn.functional as F #include relu, softmax
from einops import rearrange #for reshaping tensors (flattening)

# Enhanced Model Architecture
class SimpleNet(nn.Module): #defines neural network layer class that inherits from module
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) # conv layer of 3 rgb as input, 64 filter channel and of size 3x3
        self.bn1 = nn.BatchNorm2d(64) # this make normalization so that the mean will be 0, since if 1 activation is too large or small will affect the duration of learning model
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # model learn better when data is balanced
        
        # Second block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Third block
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # takes each channel (out of 256) and averages all its spatial values into one number.
            nn.Conv2d(256, 256//16, 1), #This reduces (squeezes) the number of channels from 256 to 16 using a 1x1 convolution.
            nn.ReLU(),
            nn.Conv2d(256//16, 256, 1), #This expands (excites) the reduced features back to 256 channels (original size).
            nn.Sigmoid() # scale 0 and 1  (~0 not immp , ~1 imp)
        )
        
        self.pool = nn.MaxPool2d(2, 2)        # Reduces spatial size by half (e.g., 32x32 → 16x16)
        self.dropout = nn.Dropout(0.3)        # Regular dropout for intermediate layers Prevents overfitting, Forces the model to not rely too much on any one neuron
        self.dropout2 = nn.Dropout(0.5)       # Stronger dropout before final FC layer, more powerful since here overfitting is more likely to have ...

        
        # Global Average Pooling instead of large FC layers
        self.global_pool = nn.AdaptiveAvgPool2d(1) #Reduces the number of parameters, overfitting
        self.fc = nn.Linear(256, num_classes) #Takes the 256 features output by global pooling and maps them to num_classes outputs.
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x))) # Conv1 -> BN -> ReLU , extract feature, normalize , activate
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        x = self.dropout(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 16x16 -> 8x8
        x = self.dropout(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Attention mechanism Compute attention weights per channel
        att = self.attention(x)
        x = x * att #Scale feature channels by attention, multiply the original features with those scores.
        
        # Global pooling and classification
        x = self.global_pool(x)  # 4x4 -> 1x1   reduce size, instead of looking at each pixel of image just take avg of it
        x = x.view(x.size(0), -1) #	Flatten tensor for classifier,  batch size, calculate this dimension so that the total number of elements stays the same.
        x = self.dropout2(x)
        x = self.fc(x) #	Output final class scores (logits)
        
        return x