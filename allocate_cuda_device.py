#! /usr/bin/python3

import pynvml
import torch as tch

def allocate_cuda():
	return tch.device('cuda:1')
#   if tch.cuda.is_available:
#       n_cuda = tch.cuda.device_count()
#       best = 'cuda'
#       if n_cuda > 1:
#           pynvml.nvmlInit()
#           best_space = 0.
#           for i in range(n_cuda):
#               gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
#               info = pynvml.nvmlDeviceGetMemoryInfo(gpu)
#               total = info.total
#               free = info.free
#               perc = free/total
#               if perc > best_space:
#                   best = f'cuda:{i}'
#                   best_space = perc
#       return tch.device(best)
#   return tch.device('cpu')

if __name__ == '__main__':
    dev = allocate_cuda()
    print(dev)
    print(dev.type)
