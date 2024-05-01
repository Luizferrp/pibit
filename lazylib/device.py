import torch
from functools import cache

@cache
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"