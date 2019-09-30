import torch
import torchvision
import torch.nn as nn

from config import Resnet


class Classifier():

    def __init__(self, num_classes):
        self.model = torchvision.models.resnet50(pretrained=True)
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < Resnet["num_layers_freeze"] + 1:
                for param in child.parameters():
                    param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def getModel(self):
        return self.model