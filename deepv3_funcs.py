#! /usr/bin/python3

from common_header import *
from module_variables import *
#from gen_nets import *
from funcs import eval_results, eval_branches
from train_funcs import train
from pandas import DataFrame
from aux_functions import *
from torch import optim, utils
import torch as tch
from itertools import repeat
import concurrent.futures as concurrent
import datetime as dttm
from torch import jit
from collections import defaultdict
import from_deepv3 as dv3

def train_deepv3(net,num_epochs,kwargs):
    #jit.enable_onednn_fusion(True)                <<<<<<<<<<<<----------------------
    #  model params.
    #n_classes = kwargs['n_classes']
    try:
        net_id = kwargs['name']
    except:
        net_id = kwargs['net_id']
    train_set = kwargs['train_set']
    val_loader = kwargs['val_loader']
    num_epochs = kwargs['num_epochs']
    device = kwargs['device'] if 'device' in kwargs.keys() else tch.device('cpu')
    use_file = kwargs['use_file'] if 'use_file' in kwargs.keys() else None
    res_dir = kwargs['mod_dir']

    ## dataset params.
    transform = kwargs['transforms'] if 'transforms' in kwargs.keys() else None
    batch_size = kwargs['batch_sizes']
    lr = kwargs['lr']
    base_lr = kwargs['base_lr'] if 'base_lr' in kwargs.keys() else None

    # define train params.
    patience = kwargs['patience'] if 'patience' in kwargs.keys() else None
    loss = kwargs['loss']#get_loss[kwargs['loss']].cuda() if device.type == 'cuda' else get_loss[kwargs['loss']]
    metrics = [(i,get_metric[i]) for i in kwargs['metrics']]  #obs.: only the first position is used for earlystoping
    train_metrics = [(i,get_metric[i]) for i in kwargs['metrics'][:2]]

    lr_law = kwargs['lr_law'] if 'lr_law' in kwargs.keys() else None
    use_scheduler = kwargs['use_scheduler'] if 'use_scheduler' in kwargs.keys() else None
    if use_scheduler:
        scheduler_patience = kwargs['s_patience'] if 's_patience' in kwargs.keys() else None
        if patience:
            scheduler_patience = int(patience*.5)

    start_from = kwargs['start_from'] if 'start_from' in kwargs.keys() else None
    if start_from:
        start_from = os.path.join(kwargs['main_dir'], start_from)
    ch_es = kwargs['ch_es'] if 'ch_es' in kwargs.keys() else None
    minimize = kwargs['minimize'] if 'minimize' in kwargs.keys() else True
    #l_criteria = kwargs['l_criteria']
    #dec_law = kwargs['dec_law']
    #-----------------------------

    if hasattr(net, 'n_branches'):
        n_branches = net.n_branches
    else:
        n_branches = None

    #customize the lines below
    if n_branches and base_lr:
        params = [{'params': net.base_model.parameters(), 'lr': base_lr},
                  {'params': net.classifier.parameters(), 'lr': lr*1.1}]    #no_skip_2 1.1
        params.append({'params': net.branches.parameters(), 'lr': lr})
        optimizer = optim.SGD(params, lr=lr, momentum=.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)

    time = dttm.datetime.now().strftime('%m/%d %H:%M:%S')
    msg = f'--> Started training {net_id} (time: {time})\n'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg)
    else:
        print(msg)

    saveat = os.path.join(res_dir,f'{net_id}.pth')
    save_model = kwargs['save_model'] if 'save_model' in kwargs.keys() else saveat[:-4] + 'final.pth'
    net_res = None
    if transform:
        transform = jit.script(transform)
    for b_size in batch_size if isinstance(batch_size, list) else [batch_size]:
        time = dttm.datetime.now().strftime('%H:%M:%S')
        msg = f'<< {net_id} progress update >> B. Size: {b_size}; time: {time}\n'
        if use_file:
            with open(use_file, 'a') as f:
                f.write(msg)
        else:
            print(msg)

        #define number of workers and prefetch factor
        num_workers = kwargs['def_nworkers'](b_size)*tch.cuda.device_count()
        p_factor = kwargs['def_prefetch'](b_size)

        #change lr
#       if lr_law:
#           if isinstance(lr[0], list) or isinstance(lr[0],tuple):
#               if optimizer.param_groups[0]['lr'] != lr[lr_law(b_size)][0]:
#                   optimizer.param_groups[0]['lr'] = lr[lr_law(b_size)][0]
#                   optimizer.param_groups[1]['lr'] = lr[lr_law(b_size)][1]
#           else:
#               if optimizer.param_groups[0]['lr'] != lr[lr_law(b_size)]:
#                   optimizer.param_groups[0]['lr'] = lr[lr_law(b_size)]
        if use_scheduler:       #ajeitar scheduler
            if scheduler_patience:
                if not base_lr:
                    min_lr = lr*.01
                else:
                    min_lr = [base_lr*.01 for _ in range(len(params)-1)] + [lr*.01]        #gambiarra?
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.75,
                                    mode='min' if minimize else 'max',
                                    patience=scheduler_patience, eps=1e-6, min_lr=min_lr)
            else:
                min_lr = 0
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda k: (1 - k/num_epochs)**.9)
            ret_lr=True

        #define data loader
        train_loader = utils.data.DataLoader(train_set, batch_size=b_size,
                                            shuffle=True, num_workers=num_workers,
                                            drop_last=False, prefetch_factor=p_factor,
                                            pin_memory=True)

        if ch_es and ch_es(b_size):
            metrics.reverse()

        #train
        aux = train(net, train_loader, loss, val_iter=val_loader, num_epochs=num_epochs,
                    updater=optimizer, patience=patience, saveat=saveat,
                    start_from=start_from if start_from else None, device=device,
                    #start_from=saveat if (start_from or net_res) else None, device=device,
                    use_file=use_file, verbose=True, metrics=train_metrics, name=net_id,
                    scheduler=scheduler, min_lr=min_lr, ret_lr=ret_lr, minimize=minimize,
                    n_branches=n_branches, nout_channels=kwargs['nout_channels'])#, l_when=l_criteria[0]+l_criteria[1]*i)

        #concatentate dict lists
        net_res = {key: value + aux[key] for key,value in net_res.items(V)} if net_res else aux
        del train_loader


    #save res --- good for plotting
    save_res = os.path.join(res_dir,f'{net_id}_tr.csv')
    DataFrame.from_dict(net_res).to_csv(save_res, index=False)

    #saving trained model
    save_dict = tch.load(saveat)
    net.load_state_dict(save_dict['model_state_dict'])
    tch.save(net, save_model)
    time = dttm.datetime.now().strftime('%m/%d %H:%M:%S')
    msg = f'--> Finished training {net_id} (time: {time})\n'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg)
    else:
        print(msg)

    return save_model

#function to set up multiprocessing training
def eval_deepv3(kwargs):
    og_dir = os.getcwd()
    res_dir = kwargs['res_dir']
    device = kwargs['device']

    use_file = kwargs['use_file'] if 'use_file' in kwargs.keys() else None
    name = kwargs['name']
    saveat = os.path.join(res_dir, name)
    kwargs['mod_dir'] = saveat
    save_res = os.path.join(saveat, name + '_res.csv')

    try:
        os.makedirs(saveat)
    except:
        pass


    n_branches = kwargs['n_branches']
    base_model = os.path.join(saveat, name + '_base.pth')
    with tch.no_grad():
        net = dv3.branchyDeepv3(base_model, f'deeplabv3_{type}', n_branches, kwargs['input_dim'], count_branches = kwargs['count_branches'], skip=kwargs['skip'])  if n_branches else dv3.get_base_model(base_model, f'deeplabv3_{type}')

    if n_branches != net.n_branches:
        n_branches = net.n_branches
        kwargs['loss'].update_n(n_branches)
        kwargs['n_branches'] = n_branches
        msg = f'<< {name} progress update >> Number of branches is different then antecipated: {n_branches} branches\n'
        if use_file:
            with open(use_file, 'a') as f:
                f.write(msg)
        else:
            print(msg)
    final_model = os.path.join(saveat, name + '.pth')

    try:
        num_epochs = kwargs['num_epochs']
    except:
        num_epochs = 0

    if num_epochs:
        val_set = kwargs['val_set']
        val_loader = tch.utils.data.DataLoader(val_set, batch_size=5, shuffle=False,
                                            num_workers=4, drop_last=False, prefetch_factor=4,
                                            pin_memory=True)
        kwargs |= {'val_loader': val_loader, 'save_model': final_model}
        final_model = train_deepv3(
                    net,
                    num_epochs,
                    kwargs
                )
        net = tch.load(final_model)
    else:
        tch.save(net, final_model)

    #eval model
    net.to(device)
    net.eval()
    test_set = kwargs['test_set']
    test_loader = tch.utils.data.DataLoader(test_set, batch_size=5, shuffle=False,
                                        num_workers=4, drop_last=False, prefetch_factor=4,
                                        pin_memory=True)
    aux_res = mIoU_evaluator(net, n_branches + 1, kwargs['nout_channels'], test_loader, device)    #21 voc classes

    res = defaultdict(list)
    res['net_id'].append(name)
    for key,val in aux_res.items():
        res[key].append(val)

    mIoU_res = f'./mIoU_{n_branches}_branches_results.csv'
    DataFrame.from_dict(res).set_index('net_id').to_csv(mIoU_res, mode='a',
                                                    header=not os.path.exists(mIoU_res))

    return final_model
