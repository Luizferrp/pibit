import torch
import numpy as np
class ToImage:

    def __call__(self, array: torch.Tensor, keep_normalization=True):
        """ Convert a 1d array to a 2d array by resizing (with padding) to the square root of the 1d shape

        Ex: 
            - shape: 1024  
            - sqrt(shape) = 32 -> when need round to ceil 
            - resize the feature vector to 32x32 
            - return the new feature vector as a RGB PIL Image for torchvision transforms
         """
        n = int(np.ceil((array.shape[0] ** 0.5)*2))

        array = array.cpu().numpy().copy()        
        array.resize((n, n))

        return torch.Tensor(array.astype(np.float32)).unsqueeze(0) if keep_normalization else (array * 255).astype(np.uint8)