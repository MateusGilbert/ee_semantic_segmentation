#! /usr/bin/python3

from torch import nn
import torch.nn.functional as F
import torch as tch
from new_seg_losses import SegLoss

class SegMetric(SegLoss):
    def __init__(self, smooth=1e-6, reduction='mean', avg='macro'):
        super(SegMetric, self).__init__(smooth, reduction)
        self.avg = avg

    def _compute_basics(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        n_targets = tch.max(targets.unique()).to(tch.int64).item()
        probs = tch.argmax(F.softmax(y_pred, 1).view(N,C,-1), dim=1)
        probs = F.one_hot(probs.to(tch.int64), num_classes=C).transpose(1,2)
        targets = F.one_hot(targets.view(N,-1).to(tch.int64),
                            num_classes=max(n_targets+1,C)).transpose(1,2)

        #ignore label
        if n_targets + 1 > C:
            targets = targets[:,:C,:]

        TP = (probs * targets).sum(dim=-1)
        FP = (probs * (1 - targets)).sum(dim=-1)
        FN = ((1 - probs) * targets).sum(dim=-1)
        return TP, FP, FN

class Recall(SegMetric):
    def _compute_loss(self, y_pred, targets):
        TP, _, FN = self._compute_basics(y_pred, targets)

        if self.avg == 'macro':
            return ((TP + self.smooth) / (TP + FN + self.smooth)).mean(dim=-1)
        if self.avg == 'micro':
            TP = TP.sum(dim=-1)
            FN = FN.sum(dim=-1)
        return (TP + self.smooth) / (TP + FN + self.smooth)

class Precision(SegMetric):
    def _compute_loss(self, y_pred, targets):
        TP, FP, _ = self._compute_basics(y_pred, targets)

        if self.avg == 'macro':
            return ((TP + self.smooth) / (TP + FP + self.smooth)).mean(dim=-1)
        if self.avg == 'micro':
            TP = TP.sum(dim=-1)
            FP = FP.sum(dim=-1)
        return (TP + self.smooth) / (TP + FP + self.smooth)

class  F_beta(SegMetric):
    def __init__(self, beta=1, smooth=1e-6, reduction='mean', avg='macro'):
        super(F_beta,self).__init__(smooth, reduction, avg)
        self.beta = beta

    def _compute_loss(self, y_pred, targets):
        TP, FP, FN = self._compute_basics(y_pred, targets)

        if self.avg == 'macro':
            return (((1 + self.beta**2)*TP + self.smooth)/((1 + self.beta**2)*TP + self.beta**2*FN + FP + self.smooth)).mean(dim=-1)
        if self.avg == 'micro':
            TP = TP.sum(dim=-1)
            FP = FP.sum(dim=-1)
            FN = FN.sum(dim=-1)
        return (((1 + self.beta**2)*TP + self.smooth)/((1 + self.beta**2)*TP + self.beta**2*FN + FP + self.smooth))

class Accuracy(SegMetric):
    def _compute_loss(self, y_pred, targets):
        N,C = y_pred.shape[:2]
        pred = tch.argmax(F.softmax(y_pred, 1).view(N,C,-1), dim=1)
        g_truth = targets.view(N,-1)

        acc = tch.sum(g_truth == pred,dim=1)/(g_truth.size()[1])

        return acc

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
                [0, 3, 3, 3, 3, 10]
                ],
            ],
            ])

    y_pred = 1000*tch.Tensor([
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
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
                ],
                [
                [0, 1, 1, 1, 0, 2],
                [0, 1, 0, 0, 1, 2],
                [0, 1, 0, 0, 1, 2],
                [0, 1, 1, 1, 1, 2]
                ],
            ],
            ])

    print('Accuracy')
    print(Accuracy(reduction=None)(y_pred, y_true))
    print(Accuracy()(y_pred, y_true))

    print('Macro')
    print(Recall()(y_pred, y_true))
    print(Precision()(y_pred, y_true))
    print(F_beta()(y_pred, y_true))

    print()
    print('Micro')
    print(Recall(avg='micro')(y_pred, y_true))
    print(Precision(avg='micro')(y_pred, y_true))
    print(F_beta(avg='micro')(y_pred, y_true))

    print()
    print('Check F1')
    r = Recall(reduction=None,avg=None)(y_pred, y_true)
    p = Precision(reduction=None,avg=None)(y_pred, y_true)

    print((2*p*r/(p + r)).mean(dim=-1).mean())
