#! /usr/bin/python3

from copy import deepcopy
import torch as tch
import numpy as np
#from tqdm import tqdm
import re
from module_variables import *
from funcs import *
from collections import defaultdict

def train_epoch(net, train_iter, loss, updater,
        device=tch.device('cpu')):
    if isinstance(net, tch.nn.Module):
        net.train()

#    metric = Accumulator(2)
    for X,y in train_iter:
        X,y = X.to(device,non_blocking=True),y.to(device,non_blocking=True)

        #compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat,y) #last is ground-truth
        if isinstance(updater,tch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            #Using custom buit optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
#        metric.add(float(l.cpu().sum()), y.cpu()[0])#.numel())
        del X,y

#    return metric[0]/metric[1]

def ae_train_epoch(net, train_iter, loss, updater,
                device=tch.device('cpu'), transform=None):
    if isinstance(net, tch.nn.Module):
        net.train()

    metric = Accumulator(2)
    #for X,_ in tqdm(train_iter):
    for X,_ in train_iter:
        X = X.to(device, non_blocking=True)

        #compute gradients and update parameters
        y_hat = net(transform(X) if transform else X)
        l = loss(y_hat,X.detach())
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), X.numel())

        del X,y_hat
    del l

    return metric[0]/metric[1]

def train(net, train_iter, loss, num_epochs, updater,
            val_iter=None, metrics=None, patience=None,
            saveat=None, start_from=None, verbose=False,
            device='cpu', scheduler=None, use_file=None,
            up_updater=False, ret_lr=False, ae_train=False,
            transform=None, name=None, minimize=True, start_counting=0, **kwargs):#, l_when=None):
    if verbose:
        import time

    follow = f'val_{metrics[0][0]}'  #conferir   #fazer uma média no caso branchy
    tracker = defaultdict(list)

#    if 'cuda' == device.type:              #check numb of available gpus
#        if tch.cuda.device_count() > 1:
#            net = nn.DataParallel(net)
    net.to(device)

    name = name or 'unspecified'
    counter = 0
    best_val = np.inf if minimize else 0.
    saveat = saveat or os.path.join('.','model.pth')
    if patience:
        if verbose:
            msg = f'<< {name} progress update >> Earlystopping will follow {follow} with patience set to {patience}. ae_train = {ae_train}'
            if use_file:
                with open(use_file,'a') as f:
                    f.write(msg + '\n')
            else:
                print(msg)
    else:
        patience = None
        if verbose:
            msg = f'<< {name} progress update >> Earlystopping not set.'
            if use_file:
                with open(use_file,'a') as f:
                    f.write(msg + '\n')
            else:
                print(msg)

    if start_from:
        save_dict = tch.load(start_from)
        net.load_state_dict(save_dict['model_state_dict'])
        if up_updater:
            #falta para quando tiver classificador
            lr_aux = updater.param_groups[0]['lr']
            updater.load_state_dict(save_dict['opt_state_dict'])
            updater.param_groups[0]['lr'] = lr_aux
            del lr_aux
        if patience and follow in save_dict.keys():
            best_val = save_dict[follow]# if patience else None   #see above

    epoch = 0
    #cooldown = 0
    num_epochs = num_epochs or np.inf
    tch.backends.cudnn.benchmark = True
    # change precision, allow 32 precision
    # faster, errors are greater (due to rouding)
    tch.backends.cuda.matmul.allow_tf32 = True
    tch.backends.cudnn.allow_tf32 = True
    #tch.set_float32_matmul_precision('high')                    <<<<<<<<<<<<<--------------------------
    #compiled_net = tch.compile(net)
    if 'n_branches' in kwargs.keys() and kwargs['n_branches']:
        branchy = True
        evaluator = eval_branches(kwargs['n_branches'])
    else:
        branchy = False
        evaluator = eval_results(count_one=True)#,ae_mode=ae_train)        #talvez jogar para fora
    last_lr = 0
    while True:
        epoch += 1
        if (epoch >= num_epochs):
            break

        if branchy:
            cur_lr = updater.state_dict()['param_groups'][-1]['lr']
        else:
            cur_lr = updater.state_dict()['param_groups'][0]['lr']
        #train epoch
        if verbose:
            start = time.perf_counter()
            msg = f'<< {name} progress update >> starting #{epoch} training epoch; lr = {cur_lr}, no updates since {counter} epochs\n'
            if use_file:
                with open(use_file,'a') as f:
                    f.write(msg)
            else:
                print(msg)

        if ae_train:
            #train_metrics = ae_train_epoch(compiled_net, train_iter, loss, updater, device, transform)
            train_metrics = ae_train_epoch(net, train_iter, loss, updater, device, transform)
        else:
            #train_metrics = train_epoch(compiled_net, train_iter, loss, updater, device)
            train_epoch(net, train_iter, loss, updater, device)
            #train_metrics = train_epoch(net, train_iter, loss, updater, device)

        if verbose:
            end = time.perf_counter() - start
            end_min = end // 60
            end -= end_min*60
            msg = f'<< {name} progress update >> finished #{epoch} training epoch after {end_min} mins and {end:.2f} s\n'
            if use_file:
                with open(use_file,'a') as f:
                    f.write(msg)
            else:
                print(msg)
        #tracker['train_loss'].append(train_metrics)#[0])

        if val_iter:
            with tch.no_grad():
                for met, f in metrics:
                    if met == 'mIoU':
                        cur_res = f(net, net.n_branches + 1 if branchy else 1, kwargs['nout_channels'], val_iter, device)
                    if branchy:
                        if met != 'mIoU':
                            cur_res = evaluator(net, val_iter, f, device)
                        for key,value in cur_res.items():
                            tracker[f'val_{met}_{key}'].append(value)
                        del cur_res
                    elif met == 'mIoU':
                        tracker[f'val_{met}'].append(cur_res['mIoU'])
                    else:
                        tracker[f'val_{met}'].append(evaluator(net, val_iter, f, device))

        if ret_lr or scheduler:
            tracker['lr'].append(cur_lr)#updater.state_dict()['param_groups'][0]['lr'])

        if branchy:
            with tch.no_grad():
                branch_val = [tracker[key][-1] for key in tracker.keys() if re.search(follow, key)]  #conferir
                if 'max2min' in kwargs.keys() and kwargs['max2min']:
                    weights = np.arange(len(cur_val)) + 1
                    w_max = np.max(weights)
                    if kwargs['max2min']:
                        weights = np.flip(weights)
                    cur_val = np.average(branch_val, weights=weights/w_max)
                else:
                    cur_val = np.average(branch_val)
        else:
            cur_val = tracker[follow][-1]

        if scheduler:
            scheduler.step()#cur_val)                 #reduceonplateu, change to contemplate others

        #check earlystoping criteria, if patience != None
        if patience:
            if counter < patience:
                if best_val > cur_val if minimize else best_val < cur_val:
                    #save parameters
                    save_dict = {"model_state_dict": deepcopy(net.state_dict()),
                                "opt_state_dict": deepcopy(updater.state_dict()),
                                "epoch": epoch}
                    for key,_ in metrics:
                        for k in tracker.keys():
                            if re.search(k, key):
                                save_dict[f'val_{k}'] = tracker[f'val_{k}'][-1]

                    tch.save(save_dict,saveat)
                    best_val = cur_val
                    counter = 0

                    if verbose:
                        msg = f'<< {name} progress update >> saved @ {epoch} epoch. Best score: {best_val:.5g}\n'
                        if branchy:
                            msg += 'For each branch:\n\t' + '\n\t'.join([f'b{i + 1} = {val:.5g}' for i,val in enumerate(branch_val)])
                            msg += '\n'
                        if use_file:
                            with open(use_file,'a') as f:
                                f.write(msg)
                        else:
                            print(msg)
                elif 'lr' in tracker and last_lr != cur_lr:
                    counter = 1
                    last_lr = cur_lr
                else:
                    counter += 1
            elif epoch > start_counting:
                    break
            else:
                if 'lr' in tracker and last_lr != cur_lr:           #talvez tenha um jeito mais elegante
                    counter = 0
                    last_lr = cur_lr
                counter += 1
        else:
            if best_val > cur_val if minimize else best_val < cur_val:
                #save parameters
                save_dict = {"model_state_dict": deepcopy(net.state_dict()),
                            "opt_state_dict": deepcopy(updater.state_dict()),
                            "epoch": epoch}
                for key,_ in metrics:
                    for k in tracker.keys():
                        if re.search(k, key):
                            save_dict[f'val_{k}'] = tracker[f'val_{k}'][-1]

                tch.save(save_dict,saveat)
                best_val = cur_val
                counter = 0
                if verbose:
                    msg = f'<< {name} progress update >> saved @ {epoch} epoch. Best score: {best_val:.5g}\n'
                    if branchy:
                        msg += 'For each branch:\n\t' + '\n\t'.join([f'b{i + 1} = {val:.5g}' for i,val in enumerate(branch_val)])
                        msg += '\n'
                    if use_file:
                        with open(use_file,'a') as f:
                            f.write(msg)
                    else:
                        print(msg)
            else:
                counter += 1

    return tracker
