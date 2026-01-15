import torch.nn as nn
import torch.nn.functional as F

from implementation.dropout import Dropout
from nn_models.base import BaseDropoutModel

class DropoutVGG(BaseDropoutModel):
    def __init__(self, dropout_rate=0.4, dropout_mode="inverted"):
        super().__init__(
            name="VGG Dropout Neural Network",
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: 14x14
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: 7x7
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout1 = Dropout(p=dropout_rate, mode=dropout_mode)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = Dropout(p=dropout_rate, mode=dropout_mode)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x