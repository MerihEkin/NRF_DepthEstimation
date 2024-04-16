import torch.nn as nn
import torch.nn.functional as F


class ConvNet_TopOneThird(nn.Module):
    def __init__(self, window_size=150):
        super(ConvNet_TopOneThird, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_out = int((((window_size-3+2*1)/1)+1)/2)
        self.conv2_out = int((((self.conv1_out-3+2*1)/1)+1)/2)
        self.fc1 = nn.Linear(32 * self.conv2_out * self.conv2_out, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * self.conv2_out * self.conv2_out)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(x)
        return x


class ConvNet_LowerOneThird(nn.Module):
    def __init__(self, window_size=150):
        super(ConvNet_LowerOneThird, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_out = int((((window_size-3+2*1)/1)+1)/2)
        self.conv2_out = int((((self.conv1_out-3+2*1)/1)+1)/2)
        self.fc1 = nn.Linear(32 * self.conv2_out * self.conv2_out, 1)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * self.conv2_out * self.conv2_out)
        x = F.relu(self.fc1(x))
        x = self.sig(x)
        return x
    

class ConvNet_BottomOneThird(nn.Module):
    def __init__(self, window_size=150):
        super(ConvNet_BottomOneThird, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_out = int((((window_size-3+2*1)/1)+1)/2)
        self.fc1 = nn.Linear(16 * self.conv1_out * self.conv1_out, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * self.conv1_out * self.conv1_out)
        x = F.relu(self.fc1(x))
        x = self.sig(x)
        return x