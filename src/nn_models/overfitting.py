import torch.nn as nn

from implementation.dropout import Dropout
from nn_models.base import BaseDropoutModel

# Neural network prone to overfit
class OverfittingProneNetwork(BaseDropoutModel):
    def __init__(self, dropout_rate=0.4, dropout_mode="inverted", input_channels=1, img_size=28):
        super().__init__(
            name="Overfitting-Prone Deep Network",
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            input_channels=input_channels,
            img_size=img_size
        )

        self.flatten_dim = input_channels * img_size * img_size
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(self.flatten_dim, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.dropout2 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        
        self.fc3 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        self.dropout3 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        
        self.fc4 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        
        self.fc5 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = Dropout(p=self.dropout_rate, mode=self.dropout_mode)
        
        self.fc6 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.dropout4(self.relu4(self.fc4(x)))
        x = self.dropout5(self.relu5(self.fc5(x)))
        
        x = self.fc6(x)
        return x
    
