#! /usr/bin/python3

from common_header import *
import branchy_seg_losses as BSL
import glob
from exp_setup import *
from get_seg_datasets import LoadDataset
import errno
import concurrent.futures as concurrent
from itertools import repeat
import argparse
#from aux_functions import move_file
from train_test import eval_net
from allocate_cuda_device import allocate_cuda
from deepv3_funcs import eval_deepv3
#########################################
#from torch import device
from torch.nn import ModuleList
from torchvision import transforms as tr
#########################################

#model = 'mobilenet_v2_wdil'
parser = argparse.ArgumentParser(description='Evaluate branched deepv3.')
parser.add_argument('-t', '--type', type=str, default='resnet101')
parser.add_argument('-n', '--n_branches', type=int, default=0)
parser.add_argument('-N', '--Name', type=str, default='deep_v3_resnet101')
parser.add_argument('-p', '--print_file', type=str, default=None)
parser.add_argument('-e', '--num_epochs', type=int, default=0)
parser.add_argument('-l', '--lr', type=float, default=.01)
parser.add_argument('-L', '--base_lr', type=float, default=0)
parser.add_argument('-c', '--count_branches', action='store_true')
parser.add_argument('-s', '--skip', type=int, default=0)
parser.set_defaults(count_branches=False)
args = parser.parse_args()

types = args.type
n_branches = args.n_branches
name = args.Name
num_epochs = args.num_epochs
lr = args.lr
base_lr = args.base_lr
count_branches = args.count_branches
if n_branches and not base_lr:
    base_lr = lr
skip = args.skip


dataset = 'voc_seg'
use_file = args.print_file or f'{dataset}_deepv3_msgs.txt'

og_dir = os.getcwd()
r_dir = os.path.join(og_dir,f"{dataset}_results")

try:
    os.makedirs(r_dir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

#dataset
data_path = os.path.join(og_dir,f"datasets/{dataset.split('_')[0]}")
input_dim = 256#320
target_dim = None#input_dim//8

hand_data = LoadDataset(input_dim, target_dim, None, None)

train_set, val_set, test_set = hand_data.get_dataset(data_path, dataset)#, split_ratio, idx_path=data_path, idx=idxs)

#Can't pickle <function <lambda> at 0x7f7903994430>: attribute lookup <lambda> on __main__ failed
#def _def_prefetch(x):
#    return min(10, max(2, 100 // x))
#def _def_nworkers(x):
#    return max(1, min(6, 200 // x))
    #max. suggested for this system is 6 workers
def _def_prefetch(x):
    return 4#min(10, 40 // x)
def _def_nworkers(x):
    return 4#min(6, 40 // x)
    #max. suggested for this system is 6 workers
def _lr_law(x):
    return 0 if x < 20 else 1
def _ch_es(x):#change the metric that we will follow in earlystopping when bs == 40
    return x == 40


dts_info = {
        'device': allocate_cuda(),
        'name': name,
        'main_dir': og_dir,
        'n_procs': 1,
        'n_rep': 1,
        'res_dir': r_dir,
        #dataset info
        'input_dim': input_dim,
        #'n_classes': n_classes,
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        #data fransforms
        'use_file': use_file,
        #Can't pickle <function <lambda> at 0x7f7903994430>: attribute lookup <lambda> on __main__ failed
        'def_prefetch': _def_prefetch,
        'def_nworkers': _def_nworkers, 
        'metrics': ['mIoU'], #['SSIM', 'MSE'] if ae_train else ['accuracy', 'precision', 'recall', 'Top3acc', 'Top5acc', 'F1'],
        'ch_es': None,#_ch_es,
        'minimize': False,
        'n_branches': n_branches,
        'count_branches': count_branches,
        'lr': lr,
        'base_lr': base_lr,
        'num_epochs': num_epochs,
        'batch_sizes': 10,
        'loss': BSL.LovaszSoftmax(classes='present', ignore=21, n_branches=n_branches, prev_out=True),
        'use_scheduler': True,
        'nout_channels': 21,
        'skip': skip,
        #'lr_law': _lr_law,
    }

#kwargss = list(map(get_info,models,repeat(dts_info)))

##multiprocessing for all networks
#with concurrent.ProcessPoolExecutor(max_workers=2) as executor:
#    res_dir = executor.map(eval_net,kwargss)
ret = eval_deepv3(dts_info)

msg = f'Finished training. model is saved @ {ret}'

if use_file:
    with open(use_file,'a') as f:
        f.write(msg + '\n')
        f.write('-'*20 + '\n')
else:
    print(msg)
