import torch
from torch.utils.data import DataLoader, Dataset
from .now import now
from tqdm import tqdm
from .device import device
from torch import nn
def train(epoch: int, optimizer, loss_fn, MODEL: nn.Module, dataset: DataLoader) -> dict:
    
    qt = 1
    correct = 0
    running_loss = 0
    
    # Put the MODEL in the training mode    
    MODEL.train()

    it = tqdm(enumerate(dataset), total=len(dataset))

    for _, (x, y) in it:
        x = x.to(device())
        y = y.to(device())
                
        # Make predictions for this batch
        outputs = MODEL(x)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss = loss_fn(outputs, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        correct += torch.sum(torch.argmax(outputs, 1).eq(y)).item()
        qt += len(x)
    
        # Gather data and report
        running_loss += loss.item()
        n = now()
        it.set_description(f"[{n}] Epoch {str(epoch).zfill(3)} Acc: {correct/qt:.4f} Loss: {running_loss / len(dataset):.8f}")
    
    # Loss / Accuracy
    return {
        "epoch":epoch, 
        "time":n,
        "loss":running_loss/len(dataset), 
        "running_loss":running_loss, 
        "datas_size":len(dataset), 
        "correct":correct, 
        "qt":qt, 
        "acc":correct/qt
        } 