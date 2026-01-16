import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDropoutModel(nn.Module, ABC):
    def __init__(self, name, dropout_rate=0.4, dropout_mode="inverted", input_channels=1, img_size=28):
        super().__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode
        self.input_channels = input_channels
        self.img_size = img_size
    
    @abstractmethod
    def forward(self, x):
        pass