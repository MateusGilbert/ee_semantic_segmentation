#! /usr/bin/python3

from common_header import *
from common_torch import *
from module_variables import *
from get_seg_datasets import LoadDataset
from compute_mIoU import mIoU
from allocate_cuda_device import allocate_cuda
from funcs import eval_results, eval_branches
from collections import defaultdict
import errno
import argparse
import torchvision
import numpy as np
import skimage.measure as pool
import torch.nn.functional as F
from scipy.stats import entropy

class img_norm_entropy:
    def __init__(self, n_classes, pool_min=False, s=1):
        self.pool = s != 1
        self.pool_min = pool_min
        self.size = (s,s)
        self.C = n_classes

    def __call__(self, probs):
        p = probs.numpy()
        assert len(p.shape) == 3
        entropies = entropy(p, base=self.C, axis=0)

        #pooling
        if self.pool:
            if self.pool_min:
                return np.mean(pool.block_reduce(entropies, self.size, np.min))
            return np.mean(pool.block_reduce(entropies, self.size, np.max))
        return np.mean(entropies)

def br_evaluator(net, n_exits, n_classes, test_loader, device, tau, metric='ent', size=1, ignore=(), skip=0):
    accumulator = [mIoU(n_classes=n_classes, device=device) for _ in range(n_exits+1)]
    out_count = [0 for _ in range(n_exits+1)]

    if metric.lower() == 'max':
        l = img_norm_entropy(n_classes, s=size)
    elif metric.lower() == 'min':
        l = img_norm_entropy(n_classes, s=size, pool_min=True)
    else:
        l = img_norm_entropy(n_classes)

    n_branches = n_exits - 1
    with tch.no_grad():
        for X, y in test_loader:
            X,y = X.to(device, non_blocking=True),y.to(device, non_blocking=True)

            left = False
            y_pred = net(X)

            for i in range(skip,n_branches):
                probs = F.softmax(y_pred[i],1).squeeze()
                t = l(probs.cpu())
                if t < tau:
                    accumulator[i](y_pred[i],y)
                    accumulator[-1](y_pred[i],y)
                    out_count[i] += 1
                    left = True
                    break
            if not left:
                accumulator[-2](y_pred[-1], y)
                accumulator[-1](y_pred[-1], y)
                out_count[-2] += 1
            out_count[-1] += 1
    
    res = dict()
    for i in range(n_branches):
        res[f'b{i+1}_mIoU'] = accumulator[i].compute().item()
        res[f'b{i+1}_count'] = out_count[i]
    res['mIoU_out'] = accumulator[-2].compute().item()
    res['count_out'] = out_count[-2]
    res['mIoU_gl'] = accumulator[-1].compute().item()
    res['out_gl'] = out_count[-1]
    res['t'] = tau
    res['pool'] = metric
    res['pool_size'] = size
    del accumulator
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models.')
    parser.add_argument('-M', '--models', nargs='+', default=[])
    parser.add_argument('-c', '--n_classes', type=int, default=None)
    parser.add_argument('-D', '--dimensions', type=int, nargs='+', default=[256, 256])
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-n', '--n_branches', type=int, default=0)
    parser.add_argument('-s', '--save_at', type=str, default='sim_results')
    parser.add_argument('-m', '--metric', type=str, default='ent')
    parser.add_argument('-t', '--threshold', type=float, default=.5)
    parser.add_argument('-S', '--skip', type=int, default=0)
    parser.add_argument('-p', '--pool_size', type=int, default=1)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    models = args.models
    dataset = args.dataset
    n_classes = args.n_classes
    verbose = args.verbose
    n_branches = args.n_branches
    input_dim = tuple(args.dimensions[:2])
    if len(input_dim) < 2 or input_dim[0] == input_dim[1]:
        input_dim = input_dim[0]
    save_at = args.save_at
    metric = args.metric
    tau = args.threshold
    skip = args.skip
    pool_size = args.pool_size
    assert metric.lower() in ['ent', 'max', 'min']

    if not n_classes or n_classes < 0:
        raise Exception('Number of classes unspecified! Unnable to compute mIoU.')
        exit(1)

    og_dir = os.getcwd()
    r_dir = os.path.join(og_dir,f"{dataset}_results")

    try:
        os.makedirs(r_dir)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    #dataset
    data_path = os.path.join(og_dir,f"datasets/{dataset.split('_')[0]}")

    target_dim = None
    hand_data = LoadDataset(input_dim, target_dim, None, None)

    _, _, test_set = hand_data.get_dataset(data_path, dataset)#, split_ratio, idx_path=data_path, idx=idxs)
    test_loader = tch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                        num_workers=4, drop_last=False, prefetch_factor=2,
                                        pin_memory=True)

    res = defaultdict(list)

    device = allocate_cuda()
    for model in models:
        net_id = model.split('/')[-1][:-4]
        net = tch.load(model, map_location=tch.device('cpu'))
        print('Allocating EE-DNN to device...')
        net = net.to(device)
        print('...done!')
        net.eval()

        if verbose:
            print(f'Evaluating {net_id}...')
        res['net_id'].append(net_id)
        n_exits = n_branches + 1
        aux_res = br_evaluator(net, n_exits, n_classes, test_loader, device, tau, metric=metric, size=pool_size, ignore=(n_classes-1,), skip=skip)
        #aux_res = br_evaluator(net, n_exits, n_classes, test_loader, device, metric, tau, ignore=(0, n_classes-1), skip=skip)
        for key,val in aux_res.items():
            res[key].append(val)
        if verbose:
            print(f'... finished evaluation of {net_id}')
    save_at = os.path.join(og_dir,save_at if save_at[-3:] == 'csv' else f'{save_at}.csv')
    DataFrame.from_dict(res).set_index('net_id').fillna(0).to_csv(save_at, mode='a', header=not os.path.exists(save_at))
