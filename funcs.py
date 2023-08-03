#! /usr/ubin/python3

import torch as tch
from module_variables import *

def repeat_tensor(tensor, n):   #added n_channels
    #generated by ChatGPT
    """
    Repeats the (x, y) dimensions of a PyTorch tensor n times, creating a new tensor with shape (n_batches, n, x, y).

    Args:
    - tensor: a PyTorch tensor with shape (n_batches, x, y)
    - n: an integer specifying the number of times to repeat the (x, y) dimensions

    Returns:
    - a new PyTorch tensor with shape (n_batches, n, x, y)
    """
    # Get the shape of the input tensor
    if len(tensor.shape) == 3:
        n_batches, x, y = tensor.shape
        n_channels = None
    else:
        n_batches, n_channels, x, y = tensor.shape

    # Create a tensor of zeros with the new shape
    if n_channels:
        repeated_tensor = tch.zeros((n_batches, n, n_channels, x, y), dtype=tensor.dtype, device=tensor.device)
    else:
        repeated_tensor = tch.zeros((n_batches, n, x, y), dtype=tensor.dtype, device=tensor.device)

    # Iterate over the new dimension and repeat the (x, y) dimensions
    for i in range(n):
        if n_channels:
            repeated_tensor[:, i, :, :, :] = tensor
        else:
            repeated_tensor[:, i, :, :] = tensor

    return repeated_tensor

class Branchy_loss(nn.Module):
    def __init__(self, loss, weight='equal'):
        super(Branchy_loss, self).__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, output, target):
        #assuming the first dimension is the number of batches
        n_batches,n_branches = output.size()[:2]

        output = tch.permute(output, (1, 0, 2, 3, 4))
        loss = tch.cat([
                self.loss(
                    branch_out,
                    target).unsqueeze(0)
                for branch_out in output]
            )

        if self.weight == 'equal':
            return loss

        if self.weight == 'min_first':
            mask = tch.div(tch.arange(1, n_branches + 1), n_branches)
        elif self.weight == 'max_first':
            mask = tch.div(tch.arange(n_branches, 0, -1), n_branches)
        else:
            maks = tch.ones(n_branches)

        return mask * loss

class Accumulator:
    def __init__(self,n):
        self.data = [.0]*n
    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class eval_results:
    def __init__(self,iter_mode=False, ae_mode=False, transform=False, count_one=False):
        self.iter_mode = iter_mode
        self.ae_mode = ae_mode
        self.transform = transform
        self.count_one = count_one

    def __call__(self, net, data_iter, loss, device='cpu'):
        if isinstance(net, tch.nn.Module):
            net.eval()
        tracker = Accumulator(2)
        pred_first = issubclass(type(loss), nn.Module)
        with tch.no_grad():
            for X, y in data_iter:
                X,y = X.to(device, non_blocking=True),y.to(device, non_blocking=True)

                if self.iter_mode:
                    self.__iter_comp(#first ground truth
                                X if self.ae_mode else y,
                                net(transform(X) if self.transform else X),
                                loss,
                                tracker
                            )
                else:
                    if pred_first:
                        tracker.add(
                                loss(#nn functions, predicted comes first, sklearn it is the opposite
                                        net(transform(X) if self.transform else X),
                                        X if self.ae_mode else y,  #ground-truth
                                    ).item(),
                                # X.numel() if self.ae_mode else y.numel()
                                1 if self.count_one else y.size()[0]
                            )
                    else:
                        tracker.add(
                                loss(#nn functions, predicted comes first, sklearn it is the opposite
                                        X if self.ae_mode else y,  #ground-truth
                                        net(transform(X) if self.transform else X),
                                    ).item(),
                                # X.numel() if self.ae_mode else y.numel()
                                1 if self.count_one else y.size()[0]
                            )
        return tracker[0]/tracker[1]

    def __iter_comp(self,Y_pred,Y,loss,tracker):
        for y, y_pred in zip(Y.cpu(),Y_pred.cpu()):
            if loss == SSIM:
                tracker.add(loss(y.numpy(), y_pred.numpy(), channel_axis=0),1)
            else:
                tracker.add(loss(y.squeeze().numpy(), y_pred.numpy()),1) #first is ground-truth

#add eval branches
class eval_branches:
    def __init__(self, n, ae_mode=False, transform=False): #add count one
        self.ae_mode = ae_mode
        self.transform = transform
        self.n_branches = n

    def __call__(self, net, data_iter, loss, device='cpu'):###conferir, talvez tirar net e usar um tensor placeholder
        if isinstance(net, tch.nn.Module):
            net.eval()
        tracker = Accumulator(self.n_branches + 2)  #n_branches + out + counter
        pred_first = issubclass(type(loss), nn.Module)
        with tch.no_grad():
            for X, y in data_iter:
                X,y = X.to(device, non_blocking=True),y.to(device, non_blocking=True)
                Y_hat = net(transform(X) if self.transform else X)
                losses = list()
                for i in range(self.n_branches + 1):        #n branches + output
                    if pred_first:
                        losses.append(
                                    loss(Y_hat[i], y).item(),
                                )
                    else:
                        losses.append(
                                    loss(y, Y_hat[i]).item()
                                )
                tracker.add(*losses, 1)

        results = {f'b{i+1}': tracker[i]/tracker[-1] for i in range(self.n_branches)}
        results |= {'out': tracker[-2]/tracker[-1]}     #conferir as duas linhas

        return results

#   def __iter_comp(self,Y_pred,Y,loss,tracker):
#       for y, y_pred in zip(Y.cpu(),Y_pred.cpu()):
#           if loss == SSIM:
#               tracker.add(loss(y.numpy(), y_pred.numpy(), channel_axis=0),1)
#           else:
#               tracker.add(loss(y.numpy(), y_pred.numpy()),1) #first is ground-truth

if __name__ == '__main__':
    import torch as tch
    import numpy as np
    from seg_losses import FocalLoss
    import seg_funcs as sf

    shape = (5, 10, 10)
    target = tch.tensor([
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
#        [
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        ],
        ])#.unsqueeze(0)
    print(target)
    pred = [
        [sf._convert_true([      #branch 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 2, 2, 2, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 2, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 2, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 2, 3, 4, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 3, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 2, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ], shape=shape),
        sf._convert_true([          #branch 2
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 2, 1, 0],
            [1, 1, 2, 2, 2, 2, 2, 3, 1, 0],
            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 3, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 3, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 3, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 3, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 2, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ], shape=shape),
        sf._convert_true([          #branch 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
            [1, 1, 2, 3, 4, 4, 3, 3, 1, 0],
            [1, 1, 2, 3, 4, 4, 3, 3, 1, 0],
            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], shape=shape),
        sf._convert_true([          #branch 4
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ], shape=shape)],
#        [sf._convert_true([      #branch 1
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 2, 1, 1, 1, 1, 1, 1, 1, 0],
#            [0, 2, 2, 2, 2, 2, 2, 2, 1, 0],
#            [0, 2, 2, 2, 3, 3, 3, 2, 1, 0],
#            [0, 1, 2, 2, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 2, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 2, 3, 4, 3, 2, 1, 0],
#            [0, 1, 2, 2, 2, 3, 2, 2, 1, 0],
#            [0, 1, 1, 1, 1, 2, 1, 1, 1, 0],
#            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#        ], shape=shape),
#        sf._convert_true([          #branch 2
#            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#            [1, 1, 1, 1, 1, 1, 1, 2, 1, 0],
#            [1, 1, 2, 2, 2, 2, 2, 3, 1, 0],
#            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 3, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 3, 1, 0],
#            [0, 1, 2, 3, 3, 3, 3, 3, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 3, 1, 0],
#            [0, 1, 1, 1, 1, 1, 1, 2, 1, 0],
#            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#        ], shape=shape),
#        sf._convert_true([          #branch 3
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
#            [1, 1, 2, 3, 4, 4, 3, 3, 1, 0],
#            [1, 1, 2, 3, 4, 4, 3, 3, 1, 0],
#            [1, 1, 2, 3, 3, 3, 3, 3, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        ], shape=shape),
#        sf._convert_true([          #branch 4
#            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
#            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
#            [0, 1, 2, 3, 3, 3, 3, 2, 1, 0],
#            [0, 1, 2, 2, 2, 2, 2, 2, 1, 0],
#            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
#        ], shape=shape)],
    ]
    pred = tch.tensor(np.array(pred))
    print(pred)
    my_loss = Branchy_loss(FocalLoss())
    print(my_loss(pred, target))
    my_loss = Branchy_loss(FocalLoss(), weight='min_first')
    print(my_loss(pred, target))
    my_loss = Branchy_loss(FocalLoss(), weight='max_first')
    print(my_loss(pred, target))
