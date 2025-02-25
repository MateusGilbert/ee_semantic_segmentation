#! /usr/bin/python3

import torch as tch
from torch.nn import CrossEntropyLoss

class _cross_entropy(tch.nn.Module):
    def __init__(self, reduction='mean', ignore_index=-100):
        super(_cross_entropy, self).__init__()
        self.x_entropy_loss = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def _compute_loss(self, y_pred, targets):
        if len(targets.shape) > 3:
           targets = targets.squeeze()
        return self.x_entropy_loss(y_pred, targets) 

    def forward(self, y_pred, targets):
        return self.compute_loss(y_pred, targets)

class BrXEntropyLoss(_cross_entropy):
    def __init__(self, reduction='mean', ignore_index=-100, b_reduction='mean', n_exits=0, weights=None):
        super(BrXEntropyLoss, self).__init__(reduction,ignore_index)
        self.b_reduction = b_reduction
        self.n_exits  = n_exits

        if weights and len(weights) == n_exits:
            self.weights = tch.tensor(weights, requires_grad = True)
        else:
            self.weights = weights

    def forward(self, y_pred, targets):
        if not self.n_exits:
            return self._compute_loss(y_pred, targets)
        assert self.n_exits <= y_pred.shape[0]

        losses = list()
        for i in range(self.n_exits):
            losses.append(self._compute_loss(y_pred[i], targets).unsqueeze(0))
        losses = tch.cat(losses)
        if self.weights is not None:
            losses *= self.weights.to(losses.device)

        if self.b_reduction == 'sum':
            return losses.sum()
        if self.b_reduction == 'mean':
            return losses.mean()
        return losses


if __name__ == "__main__":
    #get interseption over union
    from compute_mIoU import mIoU

    n_exits = 10
    # Example input tensors
    N, C, x, y = 2, 3, 5, 5                    # Batch size = 2, Classes = 3, Image dimensions = 5x5
    Y_pred     = tch.randn(N, C, x, y)         # Predicted logits
    Y          = tch.randint(0, C, (N, x, y))  # Ground truth indices
    Y_br       = tch.cat([Y_pred.unsqueeze(0) for _ in range(n_exits)], 0)

    miou = mIoU(C)
    for y_pred,y_true in zip(Y_pred,Y):
        miou(y_pred.unsqueeze(0), y_true)
    print(f'mIoU             : {miou.compute()}')
    miou = mIoU(C)
    miou(Y_pred, Y)
    print(f'mIoU             : {miou.compute()}')

    x_loss = BrXEntropyLoss(n_exits=n_exits)
    loss = x_loss(Y_br, Y)
    print(f"Br XEnt Loss     : {loss.item()}")

    weights = [i+1/n_exits for i in range(n_exits)]
    x_loss = BrXEntropyLoss(b_reduction='sum', n_exits=n_exits, weights=weights)
    loss = x_loss(Y_br, Y)
    print(f"Br XEnt Loss     : {loss.item()}")
