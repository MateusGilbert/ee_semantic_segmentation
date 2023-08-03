#! /usr/bin/python3

from common_header import *
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
#########################################
#from torch import device
from torch.nn import ModuleList
from torchvision import transforms as tr
#########################################

#model = 'mobilenet_v2_wdil'
parser = argparse.ArgumentParser(description='Evaluate models. To train autoencoders, insert -a|--ae')
parser.add_argument('-r', '--recipe', nargs='+', default=[])
parser.add_argument('-p', '--print_file', type=str, default=None)
args = parser.parse_args()

models = args.recipe
dataset = 'voc_seg'
use_file = args.print_file or f'{dataset}_msgs.txt'

og_dir = os.getcwd()
r_dir = os.path.join(og_dir,f"{dataset}_results")

try:
    os.makedirs(r_dir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

models = [os.path.join(og_dir,m) for m in models]

#dataset
data_path = os.path.join(og_dir,f"datasets/{dataset.split('_')[0]}")
input_dim = 320

hand_data = LoadDataset(input_dim, None, None)

train_set, val_set, test_set = hand_data.get_dataset(data_path, dataset)#, split_ratio, idx_path=data_path, idx=idxs)

#Can't pickle <function <lambda> at 0x7f7903994430>: attribute lookup <lambda> on __main__ failed
#def _def_prefetch(x):
#    return min(10, max(2, 100 // x))
#def _def_nworkers(x):
#    return max(1, min(6, 200 // x))
    #max. suggested for this system is 6 workers
def _def_prefetch(x):
    return 3#min(10, 40 // x)
def _def_nworkers(x):
    return 6#min(6, 40 // x)
    #max. suggested for this system is 6 workers
def _lr_law(x):
    return 0 if x < 20 else 1
def _ch_es(x):#change the metric that we will follow in earlystopping when bs == 40
    return x == 40


dts_info = {
        'device': allocate_cuda(),
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
        'metrics': ['mIoU', 'Jaccard'], #['SSIM', 'MSE'] if ae_train else ['accuracy', 'precision', 'recall', 'Top3acc', 'Top5acc', 'F1'],
        'ch_es': None,#_ch_es,
        'minimize': False,
        'input_dim': input_dim,
        #'ignore_class': 21,
        #'lr_law': _lr_law,
    }

kwargss = list(map(get_info,models,repeat(dts_info)))

##multiprocessing for all networks
msg = f'Finished training. Results can be found @\n'
#with concurrent.ProcessPoolExecutor(max_workers=2) as executor:
#    res_dir = executor.map(eval_net,kwargss)
res_dir = map(eval_net, kwargss)

for r_dir in res_dir:
    msg += f'\t{r_dir}\n'

#mv models recepy to the correspondent res_dir
with concurrent.ThreadPoolExecutor() as executor:
    executor.submit(move_file, res_dir, models)

if use_file:
    with open(use_file,'a') as f:
        f.write(msg + '\n')
        f.write('-'*20 + '\n')
else:
    print(msg)
