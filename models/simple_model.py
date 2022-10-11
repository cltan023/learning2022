# Identical copies of two AlexNet models
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

__all__ = ['fcn']

class FullyConnected(nn.Module):

    def __init__(self, input_dim=28*28 , width=1024, depth=4, num_classes=10):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        self.fc = nn.Sequential()
        for i in range(depth):
            if i == 0:
                self.fc.add_module('fc{}'.format(i+1), nn.Linear(self.input_dim, self.width))
            else:
                self.fc.add_module('fc{}'.format(i+1), nn.Linear(self.width, self.width))
            if i < depth:
                self.fc.add_module('batchnorm{}'.format(i+1), nn.BatchNorm1d(self.width))
                self.fc.add_module('relu{}'.format(i+1), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(self.width, self.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        out = self.fc(x)
        return self.classifier(out), out

def fcn(**kwargs):
    return FullyConnected(**kwargs)
