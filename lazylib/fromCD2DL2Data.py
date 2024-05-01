from .CustomDataset import CustomDataset
from torch.utils.data import DataLoader, Dataset

def fromCD2DL2Data(x, y, transform, BATCH_SIZE: int = 128):
    cd = CustomDataset(subset=(x, y), transform=transform)
    return DataLoader(cd, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)