import torch
import torch.nn as nn 
import torch.nn.functional as F 


class VGGNet(nn.Module):
    def __init__(self, img_size: int, num_classes: int):
        super(VGGNet, self).__init__()
        # input image size with img_size x img_size x 3
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # ouput of conv1 + pool1
        output = ((img_size - 3)/1 + 1)/2 # output x output x 64
        
        self.conv2 = nn.Conv2d(in_channels=64, 
                               out_channels=128, 
                               kernel_size=3,
                               stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # output of conv2 + pool2
        output = ((output - 3) + 1)/2 # output x output x 128
        
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # output of conv3 + pool3
        output = ((output - 3) + 1)/2
        
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,
                               stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # output of conv4 + pool4
        output = ((output - 3) + 1)/2
        
        self.conv5 = nn.Conv2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # output of conv5 + pool5
        output = ((output - 3) + 1)/2
        
        self.fc1 = nn.Linear(in_features=output*output*512, 
                             out_features=4096)
        self.fc2 = nn.Linear(in_features=4096,
                             out_features=4096)
        self.fc3 = nn.Linear(in_features=4096,
                             out_features=num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
