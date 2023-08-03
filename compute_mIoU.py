#! /usr/bin/python3

from seg_metrics import SegMetric
import torch as tch

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

    evaluator = mIoU(n_classes=4)
    #evaluator = mIoU(n_classes=3, ignore_class=0)
    evaluator(y_pred,y_true)
    print(f'mIoU = {evaluator.compute()}')
    for _ in range(10):
        evaluator(y_pred,y_true)
    print(f'mIoU = {evaluator.compute()}')
