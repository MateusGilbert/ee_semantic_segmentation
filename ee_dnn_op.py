#! /usr/bin/python3

from from_deepv3 import branchyDeepv3
import copy
from torch import nn
from eval_flops import check_flops
import torch as tch
import numpy as np
import sim_metrics as M
import argparse
from allocate_cuda_device import allocate_cuda
import re
from collections import defaultdict
from get_seg_datasets import LoadDataset
import torch.nn.functional as F
from pandas import DataFrame
import os

#conferir
class mIoU:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.accumulator = np.array([[0 for _ in range(self.n_classes)] for _ in range(2)])

    def __call__(self,Img, Gt):
        for i in range(self.n_classes):
            gt    = tch.where(Gt == i, 1., 0.)
            img   = tch.where(Img == i, 1., 0.)
            inter = gt*img
            aux   = gt+img
            union = tch.where(aux > 1e-9, 1., 0.)
            self.accumulator[0,i] += tch.sum(inter)
            self.accumulator[1,i] += tch.sum(union)

    def compute(self):
        cIoU = self.accumulator[0,:] / self.accumulator[1,:]
        cIoU[cIoU == float('nan')] = 1.
        return np.sum(cIoU)/self.n_classes

class eval_ee_deeplabv3():
    def __init__(self, ee_model, metric, th, less_than=True, ignore=[], device=tch.device('cpu')):
        self.model     = ee_model #copy.deepcopy(ee_model)
        self.n         = self.model.n_branches
        self.ignore    = ignore
        self.metric    = metric
        self.less_than = less_than
        self.threshold = th
        self.device    = device
        self.last_br   = max([i for i in range(self.n) if i not in ignore])

    def __call__(self, X):
        output = dict()
        inp_shape = X.shape[-2:]

        main_flops         = list()
        branch_flops       = list()

        #op EE-DNN
        has_ref            = False
        Y_ref              = None

        #check if we "left" EE-DNN
        left               = False

        X = X.unsqueeze(0)
        for i in range(self.n):
            #compute backbone flops
            main_flops.append(
                    check_flops(self.model.base_model[i], X.shape[-2:], X.shape[1], self.device)
                )
            X = self.model.base_model[i](X)

            if i not in self.ignore and not left:
                #generate early exit image
                br_output = self.model.branches[i](X)
                br_output = F.interpolate(br_output, size=inp_shape, mode='bilinear', align_corners=False)
                br_output = br_output.argmax(dim=1)

                #compute branch flops
                branch_flops.append(
                    check_flops(self.model.branches[i], X.shape[-2:], X.shape[1], self.device)
                )

                if has_ref and (self.metric(Y_ref, br_output) < self.threshold if self.less_than else self.metric(Y_ref, br_output) > sel.threshold):
                    output['exit'] = br_output.cpu().squeeze()
                    output['exit_flops'] = sum(branch_flops) + sum(main_flops)       #conferir
                    output['exit_flops_2'] = sum(branch_flops[1:]) + sum(main_flops)
                    output['edge_flops'] = output['exit_flops']
                    output['edge_flops_2'] = output['exit_flops_2']
                    output['n'] = i+1
                    left = True
                else:
                    Y_ref = br_output
                    has_ref = True
            if not left and i == self.last_br:
                output['edge_flops'] = sum(branch_flops) + sum(main_flops)
                output['edge_flops_2'] = sum(branch_flops[1:]) + sum(main_flops)

        main_flops.append(
            check_flops(self.model.base_model[-1], X.shape[-2:], X.shape[1], self.device)
        )
        X = self.model.base_model[-1](X)
        main_flops.append(
                check_flops(self.model.classifier, X.shape[-2:], X.shape[1], self.device)
        )
        Y = self.model.classifier(X)
        Y = F.interpolate(Y, size=inp_shape, mode='bilinear', align_corners=False)
        Y = Y.argmax(dim=1)
        output['last']    = Y.cpu().squeeze()
        output['last_flops'] = sum(branch_flops) + sum(main_flops)              #conferir
        output['last_flops_2'] = sum(branch_flops[1:]) + sum(main_flops)              #conferir
        if not left:
            output['exit'] = Y.cpu().squeeze()
            output['exit_flops'] = output['last_flops']
            output['exit_flops_2'] = output['last_flops_2']
            output['n'] = self.n+1

        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EE-DNN.')
    parser.add_argument('-M', '--model')
    parser.add_argument('-m', '--metric')
    parser.add_argument('-t', '--threshold', type=float)
    parser.add_argument('-i', '--ignore_background', action='store_true')
    parser.add_argument('-I', '--ignore_branch', nargs='+', type=int, default=[])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--size', type=int, nargs='+', default=[256,256])
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-n', '--n_classes', type=int)
    parser.set_defaults(verbose=False)
    parser.set_defaults(ignore_background=False)
    args = parser.parse_args()

    model   = args.model
    verbose  = args.verbose
    img_size = args.size
    input_dim = img_size
    metric   = args.metric
    th        = args.threshold
    ignore_bk = args.ignore_background
    n_classes = args.n_classes
#    if ignore_bk:
#        ingore = (0, n_classes-1)
#    else:
#        ingore = (n_classes-1)

    ignore = (0, n_classes-1) if ignore_bk else (n_classes-1)
    if metric.lower() == 'ssim':
        l = M.SSIM(n_classes-1)
    elif metric.lower() == 'nmi':
        l = M.NMI
    elif metric.lower() == 'vi':
        l = M.VI(ignore=ignore)
    elif metric.lower() == 'h_xy':
        l = M.Seg_comp(ignore=ignore)
    elif metric.lower() == 'h_yx':
        l = M.Seg_comp(over=False,ignore=ignore)
    else:
        l = M.MSE

    ig_br     = args.ignore_branch
    if len(ig_br):
        ig_br = [i - 1 for i in ig_br]
        ig_br.sort()

    dataset   = args.dataset
    #n_classes = args.n_classes

    device   = allocate_cuda()
    net      = tch.load(model, map_location=tch.device('cpu'))
    net.to(device)
    net.eval()
    n_eexits = net.n_branches
    EE_DNN   = eval_ee_deeplabv3(net, l, th, ignore=ig_br, device=device)

    data_path = f"./datasets/{dataset.split('_')[0]}"

    target_dim = None
    hand_data = LoadDataset(input_dim, target_dim, None, None)
    _,_,test_set = hand_data.get_dataset(data_path, dataset)
    test_loader = tch.utils.data.DataLoader(test_set, batch_size=None, shuffle=False,
                                        num_workers=3, drop_last=False, prefetch_factor=2,
                                        pin_memory=True)

    res = defaultdict(list)
    res['net_id'].append(model)
    res['x'].append(img_size[0])
    res['y'].append(img_size[1])
    res['metric'].append(metric.lower())
    res['t'].append(th)
    if len(img_size) == 1:
        res['y'].append(img_size[0])

    tot_flops    = 0
    tot_flops_2  = 0
    edge_flops   = 0
    edge_flops_2 = 0
    n_imgs       = 0
    prog         = mIoU(n_classes)
    if verbose:
        print(f'Started EE-DNN evaluation.\n\tmodel: {model}')
    with tch.no_grad():
        for X, y in test_loader:
            if n_imgs % 50 == 0 and verbose:
                print(f'\tprocessed {n_imgs} images')
            X = X.to(device, non_blocking=True)
            outputs   = EE_DNN(X)
            tot_flops    += outputs['exit_flops']
            edge_flops   += outputs['edge_flops']
            tot_flops_2  += outputs['exit_flops_2']
            edge_flops_2 += outputs['edge_flops_2']
            n_imgs     += 1
            prog(outputs['exit'].to('cpu'), y)
            n_exit = outputs['n']
            e_label = 'out' if n_exit == n_eexits + 1 else f'e_{n_exit}'
            if e_label in res.keys():
               res[e_label][0] += 1
            else:
               res[e_label].append(1)
    global_mIoU    = prog.compute()
    global_flops   = tot_flops / n_imgs
    edge_flops     = edge_flops / n_imgs
    global_flops_2 = tot_flops_2 / n_imgs
    edge_flops_2   = edge_flops_2 / n_imgs

    for i in range(n_eexits):
        if f'e_{i+1}' not in res.keys():
            res[f'e_{i+1}'].append(0)
    if 'out' not in res.keys():
        res['out'].append(0)
    res['n_imgs'].append(n_imgs)
    res['avg_flops'].append(global_flops)
    res['edge_flops'].append(edge_flops)
    res['avg_flops_2'].append(global_flops_2)
    res['edge_flops_2'].append(edge_flops_2)
    res['mIoU'].append(global_mIoU)
    res['ig_bk'].append(ignore_bk)

    saveat = f'./ee_{n_eexits}_{metric}_lw_m2_res.csv'
    #sort keys
    res = dict(sorted(res.items()))
    DataFrame.from_dict(res).set_index('net_id').to_csv(saveat, mode='a',
                                                        header=not os.path.exists(saveat))
    if verbose:
        print('...done')
