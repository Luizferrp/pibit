from torch import nn
import torch

class AlexNet(nn.Module):
  def __init__(self, x, num_classes=2):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, x, kernel_size=7, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=1, stride=2),
        nn.Conv2d(x, x, kernel_size=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(x, x, kernel_size=2, padding=1),
        nn.ReLU(inplace=True),            
        nn.Conv2d(x, x, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),            
        nn.Conv2d(x, x, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
        
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(4608, x),
      nn.ReLU(inplace=True),            
      nn.Dropout(),
      nn.Linear(x, x),
      nn.ReLU(inplace=True),            
      nn.Linear(x, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x