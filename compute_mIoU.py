#! /usr/bin/python3

from seg_metrics import SegMetric
import torch as tch
import numpy as np

class mIoU(SegMetric):
    def __init__(self, n_classes, device='cpu'):#ignore_class=None, device='cpu'):
        super(mIoU, self).__init__()
        #self.device=device
        self.C = n_classes
        #self.ignore = ignore_class
        self.accumulator = tch.zeros((3, self.C))#.to(device)
        #self.accumulator = tch.zeros((3, self.C + 1 if isinstance(self.ignore,int) else self.C)).to(device)

    def forward(self, y_pred, targets):
        if y_pred.device != self.accumulator.device:
            self.accumulator = self.accumulator.to(y_pred.device)
        C = y_pred.shape[1]
        assert C == self.accumulator.shape[1]
        TP, FP, FN = self._compute_basics(y_pred, targets)
        TP = TP.sum(dim=0)
        FP = FP.sum(dim=0)
        FN = FN.sum(dim=0)
        self.accumulator[0] += TP
        self.accumulator[1] += FP
        self.accumulator[2] += FN

    def compute(self):
        den = self.accumulator.sum(dim=0)
        cIoU = tch.div(self.accumulator[0], den)
        #if isinstance(self.ignore, int):
        #    cIoU[self.ignore] = 0
        self.accumulator = self.accumulator.cpu()
        cIoU[cIoU == float('nan')] = 1.   #0/0 --> No true positive, no false positive or negative; not the best solution
        return (cIoU.sum()/self.C).cpu()

class img_mIoU(SegMetric):
    def __init__(self):#ignore_class=None, device='cpu'):
        super(img_mIoU, self).__init__()
        self.accumulator = [0, 0]

    def forward(self, y_pred, target):#modificar p/ adicionar as classes do pred
        #one image at a time
        if len(y_pred.shape) == 4:
            y_pred = y_pred.argmax(dim=1).squeeze()
        target = target.squeeze()
        classes = target.reshape(-1).unique()
        IoUsum = 0.
        for i in classes:
           gt      = tch.where(target == i, 1., 0.)
           pred    = tch.where(y_pred == i, 1., 0.)
           inter   = tch.sum(gt*pred)
           aux     = gt+pred
           union   = tch.sum(tch.where(aux > 1e-9, 1., 0.))
           IoUsum  += inter/union
        self.accumulator[0] += (IoUsum/classes.shape[0]).item()
        self.accumulator[1] += 1

    def compute(self):
        if self.accumulator[1] <= 0:
            return np.nan
        return self.accumulator[0] / self.accumulator[1]

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

    evaluator   = mIoU(n_classes=4)
    evaluator_2 = img_mIoU()
    #evaluator = mIoU(n_classes=3, ignore_class=0)
    evaluator(y_pred,y_true)
    print(f'mIoU       = {evaluator.compute()}')
#    for _ in range(10):
#        evaluator(y_pred,y_true)
#    print(f'mIoU = {evaluator.compute()}')
    evaluator_2(y_pred, y_true)
    print(f'image mIoU = {evaluator_2.compute()}')
