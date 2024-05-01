from torch import nn
import torch
from .InceptionNet import InceptionModule

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2, confidence_threshold:int=0.7):
        super(GoogLeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(128, 128, 96, 128, 16, 32, 32),
            InceptionModule(320, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(480, 192, 96, 208, 16, 48, 64),
            InceptionModule(512, 160, 112, 224, 24, 64, 64),
            InceptionModule(512, 128, 128, 256, 24, 64, 64),
            InceptionModule(512, 112, 144, 288, 32, 64, 64),
            InceptionModule(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(832, 256, 160, 320, 32, 128, 128),
            InceptionModule(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.exit_branches = nn.ModuleList([nn.Linear(128, 4608) for _ in range(num_exits)])
        self.final_exit = nn.Linear(128, 4608)
        self.confidence_threshold = confidence_threshold
            
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
          nn.Dropout(),
          nn.Linear(x, x),
          nn.ReLU(inplace=True),            
          nn.Dropout(),
          nn.Linear(x, x),
          nn.ReLU(inplace=True),            
          nn.Linear(x, 128)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        outputs = []
        max_probs = []
        for exit_branch in self.exit_branches:
            output = exit_branch(self.classifier(x))
            probs = F.softmax(output, dim=1)
            max_probs.append(torch.max(probs, dim=1)[0])
            outputs.append(output)
            if torch.max(max_probs[-1]) > self.confidence_threshold:
                return x
        final_output = self.final_exit(self.classifier(x))
        outputs.append(final_output)
        return x