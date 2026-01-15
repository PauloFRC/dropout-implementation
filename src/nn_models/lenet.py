import torch.nn as nn
import torch.nn.functional as F

from implementation.dropout import Dropout
from nn_models.base import BaseDropoutModel

class DropoutLeNet(BaseDropoutModel):
    def __init__(self, dropout_rate=0.4, dropout_mode="inverted"):
        super().__init__(
            name="LeNet Dropout Neural Network",
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode
        )

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x