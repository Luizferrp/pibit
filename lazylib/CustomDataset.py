import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
class CustomDataset(Dataset):
    """ Custom dataset class used for applying transforms to the features. """
    def __init__(self, subset: Tuple[torch.Tensor, torch.Tensor], transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.subset[0][index]) if self.transform else self.subset[0][index]
        y = self.subset[1][index]
        return x, y
        
    def __len__(self):
        return self.subset[0].size(0)