#! /usr/bin/python3

from torch import nn
import torch.nn.functional as F
import torch as tch
from lovaszsoftmax import lovasz_softmax

class SegLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(SegLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def _compute_loss(self, y_pred, targets):
        pass

    def forward(self, y_pred, targets):
        if self.reduction == 'mean':
            return self._compute_loss(y_pred, targets).mean()
        if self.reduction == 'mean_batchwise':
            loss = self._compute_loss(y_pred, targets)
            n = loss.dim()
            dims = list(range(1,n))
            return loss.mean(dim=dims)
        if self.reduction == 'sum_batchwise':
            loss = self._compute_loss(y_pred, targets)
            n = loss.dim()
            dims = list(range(1,n))
            return loss.sum(dim=dims)
        if self.reduction == 'sum':
            return self._compute_loss(y_pred, targets).sum()
        return self._compute_loss(y_pred, targets)

class DiceLoss(SegLoss):
    def __init__(self, smooth=1e-6, reduction='mean',index=False):
        super(DiceLoss, self).__init__(smooth,reduction)
        self.index = index

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

        num = 2*(probs * targets).sum(dim=(1, 2)) + self.smooth
        den = (probs + targets).sum(dim=(1, 2)) + self.smooth

        if self.index:
            return num/den
        return 1 - num/den

class JaccardLoss(DiceLoss):
    def __init__(self, smooth=1e-6, reduction='mean',index=False, downgrad_bg=1.):
        super(JaccardLoss, self).__init__(smooth,reduction,index)
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

        if self.index:
            return IoU

        if self.downgrad_bg:
            loss = 1 - IoU
            loss[:,0] *= self.downgrad_bg
            return loss

        return (1 - IoU).sum(dim=-1)

class TverskyLoss(SegLoss):
    def __init__(self, smooth=1e-6, alpha=.5, beta=.5, reduction='mean'):
        super(TverskyLoss, self).__init__(smooth, reduction)
        self.alpha = alpha
        self.beta = beta

    def _forward_imp(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        probs = F.softmax(y_pred, 1).view(N,C,-1)
        #probs = tch.argmax(F.softmax(y_pred, 1).view(N,C,-1), dim=1)
        #probs = F.one_hot(probs.to(tch.int64), num_classes=C).transpose(1,2)
        targets = F.one_hot(targets.view(N,-1).to(tch.int64), num_classes=C).transpose(1,2)
        TP = (probs * targets).sum(dim=-1)
        FP = (probs * (1 - targets)).sum(dim=-1)
        FN = ((1 - probs) * targets).sum(dim=-1)

        tversky_idx = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        return 1 - tversky_idx

    def _compute_loss(self, y_pred, targets):
        return self._forward_imp(y_pred, targets)

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, smooth=1e-6, alpha=.5, beta=.5, gamma=1., reduction='mean'):
        super(FocalTverskyLoss, self).__init__(smooth, alpha, beta, reduction)
        self.gamma = gamma

    def _compute_loss(self, y_pred, targets):
        tversky_loss = self._forward_imp(y_pred, targets)

        return tversky_loss**(1/self.gamma)

class FocalLoss(SegLoss):
    def __init__(self, alpha=None, gamma=2, smooth=1e-6, reduction='mean'):
        super(FocalLoss, self).__init__(smooth,reduction)

        self.alpha = alpha
        self.gamma = gamma

    def _compute_loss(self, y_pred, targets):
        N, C = y_pred.shape[:2]
        log_probs = F.log_softmax(y_pred, dim=1)
        targets = targets.to(tch.int64)
        ce_loss = F.nll_loss(log_probs.view(N,C,-1), targets.view(N,-1))
        probs = tch.exp(log_probs)
        pt = probs.gather(1, targets).squeeze(1)
        #loss = -((1 - pt) ** self.gamma) * log_probs.gather(1, targets).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce_loss#.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        return loss

class HybridFocalLoss(SegLoss):
    def __init__(self, smooth=1e-6, reduction='mean', focal_loss=FocalLoss(reduction='mean_batchwise'), ftversky_loss=FocalTverskyLoss(alpha=.7, beta=.3, gamma=4/3, reduction='mean_batchwise')):
        super(HybridFocalLoss, self).__init__(smooth,reduction)

        self.fl = focal_loss
        self.ftl = ftversky_loss

    def _compute_loss(self, y_pred, targets):
        fl_loss = self.fl(y_pred, targets)
        ftl_loss = self.ftl(y_pred, targets)

        return fl_loss + ftl_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax,self).__init__()

        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, y_pred, targets):
        return lovasz_softmax(y_pred, targets, classes=self.classes, per_image=self.per_image, ignore=self.ignore)

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
            ])

    #loss_ = FocalLoss()
    print('Mean')
    #print(FocalLoss()(y_pred, y_true))
    print(JaccardLoss()(y_pred, y_true))
    #print(TverskyLoss()(y_pred, y_true))
    #print(FocalTverskyLoss(gamma=4/3)(y_pred, y_true))

    print('Sum')
    #print(FocalLoss(reduction='sum')(y_pred, y_true))
    print(JaccardLoss(reduction='sum')(y_pred, y_true))
    #print(TverskyLoss(reduction='sum')(y_pred, y_true))
    #print(FocalTverskyLoss(gamma=4/3,reduction='sum')(y_pred, y_true))
