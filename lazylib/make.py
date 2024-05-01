from .train import train
from .ToImage import ToImage
from .preparation import preparation
from .fromCD2DL2Data import fromCD2DL2Data
from .validate import validate, register
from torchvision import transforms
import torch.optim as optim
from torch import nn
import numpy as np
import random
import torch

def transformate(t: transforms = ToImage):
    return transforms.Compose([
        t(),
    ])

def make(EPOCHS: int, BATCH_SIZE: int, LEARNING_RATE: int, csv_path: str, out_path:str, MODEL: nn.Module, SEED:int = 1701, xcol: int=1, transform: transforms=transformate()) -> tuple:
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    x_train, x_test, y_train, y_test = preparation(csv_path, xcol)

    data_test = fromCD2DL2Data(x_train, y_train, transform, 128)
    data_train = fromCD2DL2Data(x_test, y_test, transform, 128)
    
    train_m = []
    valid_m = []

    for epoch in range(1, EPOCHS+1):
        register(train(epoch, optimizer, loss_fn, MODEL, data_train), f'{out_path}-train--{epoch}')
        register(validate(epoch, loss_fn, MODEL, data_test), f'{out_path}-valid--{epoch}')
        #print(valid_m[epoch-1]["confusion_matrix"])
    
    print("Finished experiment!") 
    return MODEL