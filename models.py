import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=8, weights = None):
        super(ResNet, self).__init__()
        if weights:
            self.resnet = models.resnet18(weights = weights)
            
            for param in list(self.resnet.parameters()):
                param.requires_grad = False
            
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim = 1)