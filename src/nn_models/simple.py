import torch.nn as nn

from implementation.dropout import Dropout
from nn_models.base import BaseDropoutModel

class SimpleDropoutNetwork(BaseDropoutModel):
    def __init__(self, dropout_rate=0.4, dropout_mode="inverted"):
        super().__init__(
            name="Simple Dropout Neural Network",
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    