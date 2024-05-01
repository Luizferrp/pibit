import torch
import numpy as np
from torch import nn
from .now import now
from tqdm import tqdm
from .device import device
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, roc_curve, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
import pickle
import json


#def validate(epoch: int, loss_fn, MODEL, dataset: DataLoader, running_loss:float = 0., correct:int = 0, qt:int = 1) -> dict:
#    MODEL.eval()
#    it_eval = tqdm(enumerate(dataset), total=len(dataset))
#    ym = ([], [])
#    with torch.no_grad():
#        for _, (x, y) in it_eval:
#            x = x.to(device())
#            y = y.to(device())
#            output = MODEL(x)
#            running_loss += loss_fn(output, y).item()
#            ym[0].extend(torch.argmax(output, 1).cpu().numpy())
#            ym[1].extend(y.data.cpu().numpy())
#            correct += torch.sum(torch.argmax(output, 1).eq(y)).item()
#            qt += len(x)
#            n = now()
#            #probs.append(MODEL.predict_proba(x))
#            probs = torch.nn.functional.softmax(output, dim=1)[:, 1].cpu().numpy()
#            roc = roc_curve(y.data.cpu().numpy(), probs)
#            it_eval.set_description(f"[{n}] Epoch {str(epoch).zfill(3)} Val. Acc: {correct/qt:.4f} Val. Loss: {running_loss / len(dataset):.8f}")
#    
#    #return running_loss / len(dataset), correct/qt, metrics 
#    return {"epoch":epoch, "time":n, "output":output ,"metricsFromY":ym, "confusion_matrix":confusion_matrix(ym[0], ym[1]).ravel(), "acc":correct/qt, "val_loss":running_loss/len(dataset), "running_loss":running_loss, "roc_curve":roc}
##, "probs":probs, "roc_curve":roc_curve(all_labels[::], probs)

def validate(epoch: int, loss_fn, MODEL: nn.Module, dataset: DataLoader) -> dict:

    MODEL.eval()
    it_eval = tqdm(enumerate(dataset), total=len(dataset))
    y_true = []
    y_scores = []
    running_loss = 0.0
    correct = 0
    total = 0
    ym = ([], [])
    qt = 0

    with torch.no_grad():
        for _, (x, y) in it_eval:
            x = x.to(device())
            y = y.to(device())
            output = MODEL(x)
            loss = loss_fn(output, y)
            running_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            qt += len(x)
            y_true.extend(y.cpu().numpy())
            y_scores.extend(output[:, 1].cpu().numpy())  # Assuming output[:, 1] is probability of positive class
            ym[0].extend(torch.argmax(output, 1).cpu().numpy())
            ym[1].extend(y.data.cpu().numpy())
            n = now()
            it_eval.set_description(f"[{n}] Epoch {str(epoch).zfill(3)} Val. Acc: {correct/qt:.4f} Val. Loss: {running_loss / len(dataset):.8f}")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return dict({
            "epoch": epoch,
            "val_loss": running_loss / len(dataset),
            "val_accuracy": correct / total,
            "roc_curve": (fpr, tpr, thresholds),
            "auc": auc,
            "metricsFromY":ym,
            "confusion_matrix":confusion_matrix(y_true, ym[0])
        })

def takeConfusionMatrix(metrics: dict, n: int, name: str) -> plt:
    cm = np.array([[metrics[1][n]["confusion_matrix"][0],metrics[1][n]["confusion_matrix"][1]],[metrics[1][n]["confusion_matrix"][2],metrics[1][n]["confusion_matrix"][3]]])
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.savefig(name)
    return fig

def takeRocCurve(metrics: dict, n: int, name) -> plt:
    fpr, tpr, thresholds = metrics[1][n]["roc_curve"]
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.8f})')
    plt.plot([-0.01, 1.01], [-0.01, 1.01], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"{name}")
    plt.show()

def register(metrics: dict, name: str):
    with open(f'{name}.txt', 'wb') as file:
        pickle.dump(metrics, file)

def takemetrics(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)
