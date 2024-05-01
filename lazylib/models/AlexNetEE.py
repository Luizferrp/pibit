from torch import nn
import torch.nn.functional as F
import torch

class AlexNetEE(nn.Module):
  def __init__(self, x, num_classes=2, num_exits=3, confidence_threshold=0.7):
    super(AlexNetEE, self).__init__()
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

  def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
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