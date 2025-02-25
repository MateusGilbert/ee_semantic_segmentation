#! /usr/bin/python3

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, normalized_mutual_information as nmi, variation_of_information as vi
from skimage import data
import numpy as np
import matplotlib.pyplot as pl
from PIL import Image

def tensor_to_np(tensor):
    # Convert PyTorch tensor to NumPy array
    return tensor.detach().cpu().numpy()

class SSIM:
    def __init__(self, data_range):
        self.dr = data_range

    def __call__(self,tensor1, tensor2):
        if len(tensor1.shape) == 4:
            N,C,X,Y = tensor1.shape
            tensor1 = torch.argmax(F.softmax(tensor1, 1).view(N, C, -1), dim=1).view(X, Y)
            tensor2 = torch.argmax(F.softmax(tensor2, 1).view(N, C, -1), dim=1).view(X, Y)

        # Convert tensors to NumPy arrays
        tensor2 = tensor2.squeeze()
        array1 = tensor_to_np(tensor1)
        array2 = tensor_to_np(tensor2)

        # Ensure the arrays have the same data type
        array1 = array1.astype(np.int_)
        array2 = array2.astype(np.int_)

        # Compute SSIM
        ssim_value = ssim(array1, array2, data_range=self.dr)

        return ssim_value

def MSE(tensor1, tensor2):
    if len(tensor1.shape) == 4:
        N,C,X,Y = tensor1.shape
        tensor1 = torch.argmax(F.softmax(tensor1, 1).view(N, C, -1), dim=1).view(X, Y)
        tensor2 = torch.argmax(F.softmax(tensor2, 1).view(N, C, -1), dim=1).view(X, Y)

    # Convert tensors to NumPy arrays
    array1 = tensor_to_np(tensor1)
    array2 = tensor_to_np(tensor2)

    # Ensure the arrays have the same data type
    array1 = array1.astype(np.int_)
    array2 = array2.astype(np.int_)

    # Compute MSE
    mse_value = mse(array1, array2)

    return mse_value

def NMI(tensor1, tensor2):
    if len(tensor1.shape) == 4:
        N,C,X,Y = tensor1.shape
        tensor1 = torch.argmax(F.softmax(tensor1, 1).view(N, C, -1), dim=1).view(X, Y)
        tensor2 = torch.argmax(F.softmax(tensor2, 1).view(N, C, -1), dim=1).view(X, Y)


    # Convert tensors to NumPy arrays
    array1 = tensor_to_np(tensor1)
    array2 = tensor_to_np(tensor2)

    # Ensure the arrays have the same data type
    array1 = array1.astype(np.int_)
    array2 = array2.astype(np.int_)

    # Compute NMI
    nmi_value = nmi(array1, array2)

    return nmi_value

class VI:
    def __init__(self, ignore=()):
        self.ignore = ignore

    def __call__(self,tensor1, tensor2):
        if len(tensor1.shape) == 4:
            N,C,X,Y = tensor1.shape
            tensor1 = torch.argmax(F.softmax(tensor1, 1).view(N, C, -1), dim=1).view(X, Y)
            tensor2 = torch.argmax(F.softmax(tensor2, 1).view(N, C, -1), dim=1).view(X, Y)

        # Convert tensors to NumPy arrays
        array1 = tensor_to_np(tensor1)
        array2 = tensor_to_np(tensor2)

        # Ensure the arrays have the same data type
        array1 = array1.astype(np.int_)
        array2 = array2.astype(np.int_)

        # Compute VI
        vi_value = vi(array1, array2, ignore_labels=self.ignore)

        return np.sum(vi_value)

class Seg_comp(VI):
    def __init__(self, x_y=True, ignore=()):
        super().__init__(ignore=ignore)
        self.x_y = x_y

    def __call__(self,tensor1, tensor2):
        if len(tensor1.shape) == 4:
            N,C,X,Y = tensor1.shape
            tensor1 = torch.argmax(F.softmax(tensor1, 1).view(N, C, -1), dim=1).view(X, Y)
            tensor2 = torch.argmax(F.softmax(tensor2, 1).view(N, C, -1), dim=1).view(X, Y)

        # Convert tensors to NumPy arrays
        array1 = tensor_to_np(tensor1)
        array2 = tensor_to_np(tensor2)

        # Ensure the arrays have the same data type
        array1 = array1.astype(np.int_)
        array2 = array2.astype(np.int_)

        # Compute entropies
        vi_value = vi(array1, array2, ignore_labels=self.ignore)

        return vi_value[int(self.x_y)]

if __name__ == '__main__':
    tensor1 = torch.from_numpy(np.array(Image.open(f'./deepv3_101_7b_try_no_skip_2_images/2007_000129_b6.png')))
    tensor2 = torch.from_numpy(np.array(Image.open(f'./deepv3_101_7b_try_no_skip_2_images/2007_000129_b7.png')))

    f_ssim   = SSIM(20)                             #conferir
    var_info = VI(ignore=(0, 20))
    h_x_y    = Seg_comp(ignore=(0,20))
    h_y_x    = Seg_comp(x_y=False,ignore=(0,20))

    ssim_val = f_ssim(tensor1, tensor2)
    mse_val  = MSE(tensor1, tensor2)
    nmi_val  = NMI(tensor1, tensor2)
    vi_val   = var_info(tensor1, tensor2)
    over     = h_x_y(tensor1, tensor2)
    under    = h_y_x(tensor1, tensor2)
    print(f'SSIM  : {ssim_val}')
    print(f'MSE   : {mse_val}')
    print(f'NMI   : {nmi_val}')
    print(f'VI    : {vi_val}')
    print(f'H(X|Y): {over}')
    print(f'H(Y|X): {under}')
