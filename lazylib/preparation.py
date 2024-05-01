import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preparation(csv_path: str, col: int = 1):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, -col].values
    x = np.squeeze(df.iloc[:, :-col].values)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train = torch.Tensor(x_train)
    x_test  = torch.Tensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test  = torch.LongTensor(y_test)
    return x_train, x_test, y_train, y_test