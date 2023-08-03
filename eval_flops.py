#! /usr/bin/python3

from common_header import *
from common_torch import *
from module_variables import *
from get_seg_datasets import LoadDataset
from compute_mIoU import mIoU
from allocate_cuda_device import allocate_cuda
from funcs import eval_results, eval_branches
from pthflops import count_ops
from collections import defaultdict
import errno
import argparse

def check_flops(aux_model, img_dim, channels, device):
    if isinstance(aux_model, list):
        aux = aux_model
        aux_model = nn.Sequential(*aux_model)
    aux_model.to(device)
    aux_model.eval()
    if isinstance(img_dim, (list,tuple)):
        tensor = tch.rand(1,channels,img_dim[0],img_dim[1]).to(device)
    else:
        tensor = tch.rand(1,channels,img_dim,img_dim).to(device)
    tot_flops,_ = count_ops(aux_model, tensor, print_readable=False, verbose=False)
    return tot_flops

def count_flops(net, device, img_dim=256, channels=3):
    if isinstance(img_dim, int):
        x_dim,y_dim = img_dim,img_dim
    else:
        x_dim,y_dim = img_dim
    X = tch.rand(1, channels, x_dim, y_dim).to(device)
    count_branches = net.count_branches

    main_flops = list()
    branch_flops = list()
    for i in range(net.n_branches):
        main_flops.append(check_flops(net.base_model[i], (x_dim,y_dim), channels, device))
        X = net.base_model[i](X)
        channels,x_dim,y_dim = X.shape[1:]
        branch_flops.append(check_flops(net.branches[i], (x_dim,y_dim), channels, device))
    main_flops.append(check_flops(net.base_model[-1], (x_dim,y_dim), channels, device))
    X = net.base_model[-1](X)
    channels,x_dim,y_dim = X.shape[1:]
    branch_flops.append(check_flops(net.classifier, (x_dim,y_dim), channels, device))

    for i in range(1,len(main_flops)):
        main_flops[i] += main_flops[i-1]
    return [i+j for i,j in zip(main_flops, branch_flops)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models.')
    parser.add_argument('-M', '--models', nargs='+', default=[])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--size', type=int, nargs='+', default=[256])
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    models = args.models
    verbose = args.verbose
    img_size = args.size

    device = allocate_cuda()
    for model in models:
        res = defaultdict(list)
        if verbose:
            print(f'Evaluating {model}...')
        net_id = model.split('/')[-1][:-4]
        net = tch.load(model)
        n = net.n_branches
        net.to(device)
        net.eval()
        res['net_id'].append(net_id)
        res['x'].append(img_size[0])
        if len(img_size) == 1:
            res['y'].append(img_size[0])
            flops = count_flops(net, img_dim=img_size[0], device=device)
        else:
            res['y'].append(img_size[1])
            flops = count_flops(net, img_dim=img_size, device=device)
        for i,f in enumerate(flops):
            res[f'b{i+1}_flops'].append(f)
        saveat = f'./{n}_branches_model_flops.csv'
        DataFrame.from_dict(res).set_index('net_id').to_csv(saveat, mode='a',
                                                            header=not os.path.exists(saveat))
        if verbose:
            print('...done')
