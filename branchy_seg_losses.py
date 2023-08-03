#! /usr/bin/python3

from torch import nn
import torch.nn.functional as F
import torch as tch
from new_seg_losses import SegLoss
from lovaszsoftmax import lovasz_softmax

class BrSegLoss(SegLoss):
    def __init__(self, smooth=1e-6, reduction='mean', n_branches=0, weights=None):
        super(BrSegLoss, self).__init__(smooth, reduction)
        self.n = n_branches + 1
        if weights and len(weights) == n_branches + 1:
            self.weights = tch.tensor(weights, requires_grad = True)
        else:
            self.weights = tch.ones(self.n, requires_grad=True)

    def update_n(self, n):
        self.n = n + 1

    def _compute_loss(self, y_pred, targets):
        pass

    def forward(self, y_pred, targets):
        losses = list()
        for i in range(self.n):
            losses.append(self._compute_loss(y_pred[i], targets).unsqueeze(0))
        losses = tch.cat(losses)

        dim = list(range(1,len(losses.shape)))
        if self.reduction == 'mean':
            losses = losses.mean(dim=dim)
        elif self.reduction == 'sum':
            losses = losses.sum(dim=dim)
        else:
            return losses

        return tch.dot(self.weights.to(device=losses.device),losses)

class DiceLoss(BrSegLoss):
    def _compute_loss(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        probs = F.softmax(y_pred, 1).view(N,C,-1)
        targets = F.one_hot(targets.view(N,-1).to(tch.int64), num_classes=C).transpose(1,2)
        num = 2*(probs * targets).sum(dim=(1, 2)) + self.smooth
        den = (probs + targets).sum(dim=(1, 2)) + self.smooth

        return 1 - num/den

class JaccardLoss(BrSegLoss):
    def __init__(self, smooth=1e-6, reduction='mean', n_branches=0, downgrad_bg=1.):
        super(JaccardLoss, self).__init__(smooth,reduction,n_branches)
        self.downgrad_bg = downgrad_bg if 0 <= downgrad_bg <= 1. else 1.

    def _compute_loss(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        probs = F.softmax(y_pred, 1).view(N,C,-1)
        #get targets -- one_hot encoding may include void index
        n_targets = tch.max(targets.unique()).to(tch.int64).item()
        targets = F.one_hot(targets.view(N,-1).to(tch.int64),
                            num_classes=max(n_targets+1,C)).transpose(1,2)

        #remove void index
        if n_targets+1 > C:
            targets = targets[:,:C,:]

        intersection = (probs * targets).sum(dim=-1)#(1,2))
        total = (probs + targets).sum(dim=-1)#(1,2))
        union = total - intersection
        IoU = (intersection + self.smooth)/(union + self.smooth)

        if self.downgrad_bg:
            loss = 1 - IoU
            loss[:,0] *= self.downgrad_bg
            return loss

        return (1 - IoU).sum(dim=-1)

class TverskyLoss(BrSegLoss):
    def __init__(self, smooth=1e-6, alpha=.5, beta=.5, reduction='mean', n_branches=1, weights=None):
        super(TverskyLoss, self).__init__(smooth, reduction, n_branches, weights)
        self.alpha = alpha
        self.beta = beta

    def _forward_imp(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        #probs = F.softmax(y_pred, 1).view(N,C,-1)
        probs = tch.argmax(F.softmax(y_pred, 1).view(N,C,-1), dim=1)
        probs = F.one_hot(probs.to(tch.int64), num_classes=C).transpose(1,2)
        targets = F.one_hot(targets.view(N,-1).to(tch.int64), num_classes=C).transpose(1,2)
        TP = (probs * targets).sum(dim=-1)
        FP = (probs * (1 - targets)).sum(dim=-1)
        FN = ((1 - probs) * targets).sum(dim=-1)

        tversky_idx = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        return 1 - tversky_idx

    def _compute_loss(self, y_pred, targets):
        return self._forward_imp(y_pred, targets)

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, smooth=1e-6, alpha=.5, beta=.5, gamma=1., reduction='mean', n_branches=1, weights=None):
        super(FocalTverskyLoss, self).__init__(smooth, alpha, beta, reduction, n_branches, weights)
        self.gamma = gamma

    def _compute_loss(self, y_pred, targets):
        tversky_loss = self._forward_imp(y_pred, targets)

        return tversky_loss**self.gamma

class FocalLoss(BrSegLoss):
    def __init__(self, alpha=None, gamma=2, smooth=1e-6, reduction='mean', n_branches=1, weights=None):
        super(FocalLoss, self).__init__(smooth, reduction, n_branches, weights)

        self.alpha = alpha
        self.gamma = gamma

    def _compute_loss(self, y_pred, targets):
        N, C = y_pred.shape[:2]
        log_probs = F.log_softmax(y_pred, dim=1)
        probs = tch.exp(log_probs)
        targets = targets.to(tch.int64)
        pt = probs.gather(1, targets).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * log_probs.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        return loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None, n_branches=0, prev_out=False):
        super(LovaszSoftmax,self).__init__()

        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.n = n_branches + 1
        self.prev_out = prev_out

        if self.prev_out:
            self.weights = tch.linspace(0, 1, self.n + 1, requires_grad=True)[1:]

    def update_n(self, n):
        self.n = n + 1
        if self.prev_out:
            self.weights = tch.linspace(0, 1, self.n + 1, requires_grad=True)[1:]

    def forward(self, y_pred, targets):
        losses = list()
        for i in range(self.n):
            losses.append(lovasz_softmax(y_pred[i], targets, classes=self.classes, per_image=self.per_image, ignore=self.ignore).unsqueeze(0))
        if self.prev_out:
            if self.weights.device != losses[0].device:
                self.weights = self.weights.to(losses[0].device)
            return tch.dot(self.weights, tch.cat(losses)).sum()
        return tch.cat(losses).sum()


if __name__ == '__main__':
    y_true = tch.Tensor([
            [
                [
                [0, 1, 1, 1, 0, 0],
                [1, 1, 2, 2, 1, 1],
                [1, 1, 2, 2, 1, 1],
                [0, 1, 1, 1, 0, 0]
                ],
            ],
            [
                [
                [0, 3, 3, 3, 2, 0],
                [0, 3, 2, 2, 3, 1],
                [0, 3, 2, 2, 3, 1],
                [0, 3, 3, 3, 3, 0]
                ],
            ],
            ])

    y_pred = 100*tch.Tensor([
            [
                [
                [1, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1]
                ],
                [
                [0, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 1, 1, 1, 0, 0]
                ],
                [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
            ],
            [
                [
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1]
                ],
                [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [0, 0, 0, 0, .5, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [0, 1, 1, 1, 1.5, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0]
                ],
            ],
            ]).unsqueeze(0)
    y_pred = tch.cat([y_pred, y_pred, y_pred],dim=0)

    n_branches = y_pred.shape[0]
    print(f'There are {n_branches} branches')
    #loss_ = FocalLoss()
    print('Mean')
    print(LovaszSoftmax(n_branches=n_branches)(y_pred, y_true))
#    print(FocalLoss(n_branches=n_branches)(y_pred, y_true))
#    print(JaccardLoss(n_branches=n_branches)(y_pred, y_true))
#    print(TverskyLoss(n_branches=n_branches)(y_pred, y_true))
#    print(FocalTverskyLoss(gamma=4/3,n_branches=n_branches)(y_pred, y_true))

    print('Sum')
#    print(FocalLoss(reduction='sum',n_branches=n_branches)(y_pred, y_true))
#    print(JaccardLoss(reduction='sum',n_branches=n_branches)(y_pred, y_true))
#    print(TverskyLoss(reduction='sum',n_branches=n_branches)(y_pred, y_true))
#    print(FocalTverskyLoss(gamma=4/3,reduction='sum',n_branches=n_branches)(y_pred, y_true))
