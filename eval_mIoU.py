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

def mIoU_evaluator(net, n_exits, n_classes, test_loader, device):
#def mIoU_evaluator(net, n_exits, n_classes, ignore_class, test_loader, device):
    #accumulator = [mIoU(n_classes=n_classes-1 if ignore_class else n_classes, ignore_class=ignore_class, device=device) for _ in range(n_exits)]
    accumulator = [mIoU(n_classes=n_classes, device=device) for _ in range(n_exits)]

    n_branches = n_exits - 1
    with tch.no_grad():
        for X, y in test_loader:
            X,y = X.to(device, non_blocking=True),y.to(device, non_blocking=True)

            if isinstance(net, torchvision.models.segmentation.deeplabv3.DeepLabV3):
                y_pred = net(X)['out']
            else:
                y_pred = net(X)

            for i in range(n_branches):
                accumulator[i](y_pred[i],y)

            accumulator[-1](y_pred[-1] if n_branches else y_pred, y)
    
    res = dict()
    for i in range(n_branches):
        res[f'b{i+1}_mIoU'] = accumulator[i].compute().item()
    res['mIoU'] = accumulator[-1].compute().item()
    del accumulator
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models.')
    parser.add_argument('-M', '--models', nargs='+', default=[])
    parser.add_argument('-c', '--n_classes', type=int, default=None)
#    parser.add_argument('-i', '--ignore_class', type=int, default=None)
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-n', '--n_branches', type=int, default=0)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    models = args.models
    dataset = args.dataset
    n_classes = args.n_classes
#    ignore_class = args.ignore_class
    verbose = args.verbose
    n_branches = args.n_branches

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
    input_dim = 256

    hand_data = LoadDataset(input_dim, None, None)

    _, _, test_set = hand_data.get_dataset(data_path, dataset)#, split_ratio, idx_path=data_path, idx=idxs)
    test_loader = tch.utils.data.DataLoader(test_set, batch_size=5, shuffle=False,
                                        num_workers=4, drop_last=False, prefetch_factor=4,
                                        pin_memory=True)

    device = allocate_cuda()

    res = defaultdict(list)

    device = allocate_cuda()
    for model in models:
        net_id = model.split('/')[-1][:-4]
        #prefix = net_id.split('_')[0]
        net = tch.load(model)
        net.to(device)
        net.eval()
        #n_branches = net.n_branches if prefix == 'brunet' else 0

        if verbose:
            print(f'Evaluating {net_id}...')
        res['net_id'].append(net_id)
        n_exits = n_branches + 1
        aux_res = mIoU_evaluator(net, n_exits, n_classes, test_loader, device)
        for key,val in aux_res.items():
            res[key].append(val)
    #   accumulator = [mIoU(n_classes=n_classes, ignore_class=ignore_class, device=device) for _ in range(n_exits)]

    #   with tch.no_grad():
    #       for X, y in test_loader:
    #           X,y = X.to(device, non_blocking=True),y.to(device, non_blocking=True)

    #           y_pred = net(X)
    #           for i in range(n_branches):
    #               accumulator[i](y_pred[i],y)

    #           accumulator[-1](y_pred[-1] if n_branches else y_pred, y)
    #   
    #   for i in range(n_branches):
    #       res[f'b{i}_mIoU'].append(accumulator[i].compute().item())

    #   res[f'mIoU'].append(accumulator[-1].compute().item())
        if verbose:
            print(f'... finished evaluation of {net_id}')
    DataFrame.from_dict(res).set_index('net_id').to_csv('./mIoU_results.csv')
