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

#create loop iter for branchy nets

def _loop_iter(net_id,kwargs):
    #jit.enable_onednn_fusion(True)                <<<<<<<<<<<<----------------------
    #  model params.
    #n_classes = kwargs['n_classes']
    blk_config = kwargs['blk_config'] if 'blk_config' in kwargs.keys() else None
    nin_channels = kwargs['nin_channels']
    nout_channels = kwargs['nout_channels']
    ignore_class = kwargs['ignore_class'] if 'ignore_class' in kwargs.keys() else None
    conv_blk = kwargs['conv_blk']
    conv_layer = get_conv[kwargs['conv_layer']] if 'conv_layer' in kwargs.keys() else nn.Conv2d
    dilation = kwargs['dilation'] if 'dilation' in kwargs.keys() else 1
    #classifier = kwargs['classifier']
    use_res = kwargs['use_res'] if 'use_res' in kwargs.keys() else False
    net_const = kwargs['net_const']
    train_set = kwargs['train_set']
    val_loader = kwargs['val_loader']
    test_loader = kwargs['test_loader']
    num_epochs = kwargs['num_epochs']
    act_func = kwargs['act_func']
    bias = kwargs['bias']
    device = kwargs['device'] if 'device' in kwargs.keys() else tch.device('cpu')
    initializer = get_initializer[kwargs['initializer']] if 'initializer' in kwargs.keys() else None

    add_parameters = dict()         #for unet-based classifiers
    for par in ['out_dim', 'exp_factor', 'base_net', 'n_classes', 'nencout_channels']:
        if par in kwargs.keys():
            if par == 'n_classes':
                add_parameters[par] = kwargs[par]+1
            else:
                add_parameters[par] = kwargs[par]

    if 'base_net' in add_parameters.keys() and 'n_classes' not in add_parameters.keys():
        add_parameters |= {
                            'n_classes': nout_channels,
                            'nencout_channels': nout_channels,
                            'in_dim': kwargs['input_dim'],
                         }

    #----------------------------
    #       train config
    #----------------------------
    use_file = kwargs['use_file'] if 'use_file' in kwargs.keys() else None
    res_dir = kwargs['mod_dir']
    ## dataset params.
    transform = kwargs['transforms'] if 'transforms' in kwargs.keys() else None
    batch_size = kwargs['batch_sizes']
    lr = kwargs['lrs']
    # define train params.
    patience = kwargs['patience'] if 'patience' in kwargs.keys() else 0
    start_counting = kwargs['st_counting'] if 'st_counting' in kwargs.keys() else 0
    loss = get_loss[kwargs['loss']].cuda() if device.type == 'cuda' else get_loss[kwargs['loss']]
    metrics = [(i,get_metric[i]) for i in kwargs['metrics']]  #obs.: only the first position is used for earlystoping
    train_metrics = [(i,get_metric[i]) for i in kwargs['metrics'][:2]]
    lr_law = kwargs['lr_law'] if 'lr_law' in kwargs.keys() else None
    use_scheduler = kwargs['use_scheduler'] if 'use_scheduler' in kwargs.keys() else None
    if use_scheduler:
        scheduler_patience = kwargs['s_patience'] if 's_patience' in kwargs.keys() else int(patience*.5)
    start_from = kwargs['start_from'] if 'start_from' in kwargs.keys() else None
    if start_from:
        start_from = os.path.join(kwargs['main_dir'], start_from)
    ae_train = kwargs['ae_train'] if 'ae_train' in kwargs.keys() else False
    ch_es = kwargs['ch_es'] if 'ch_es' in kwargs.keys() else None
    minimize = kwargs['minimize'] if 'minimize' in kwargs.keys() else True
    #l_criteria = kwargs['l_criteria']
    #dec_law = kwargs['dec_law']
    #-----------------------------

    net = net_const(nin_channels=nin_channels,
                    blk_config=blk_config,
                    nout_channels=nout_channels,
                    conv_block=conv_blk, dilation=dilation,
                    use_res=use_res,
                    conv_layer=conv_layer,
                    bias=bias,
                    act_func=act_func,
                    **add_parameters
                ).to(device)
    if hasattr(net, 'n_branches'):
        n_branches = net.n_branches
    else:
        n_branches = None

    if initializer and not (start_from or 'base_net' in add_parameters.keys()):
        net.init_layers(initializer)

    #customize the lines below
    if hasattr(net, 'base_net') and isinstance(lr,list):
        if isinstance(lr[0], list):
            pass
        else:
            #costumize line bellow to match branchy network
            if n_branches:
                params = [{'params': net.enc_block.parameters(), 'lr': lr[0]},
                          {'params': net.enc_out.parameters(), 'lr': lr[0]}]
                if 'dec_block' in net.base_net:
                    params.extend(
                                [{'params': net.dec_block.parameters(), 'lr': lr[0]},
                                {'params': net.out_layer.parameters(), 'lr': lr[0]}]           #mesmo lr para a saída
                            )
                else:
                    params.append(
                                {'params': net.classifier.parameters(), 'lr': lr[0]}
                            )
                params.append({'params': net.branches.parameters(), 'lr': lr[1]})
                optimizer = optim.NAdam(params, lr=lr[1])

            else:
                #costumize line bellow to match classifier network
                optimizer = optim.NAdam([{'params': net.enc_block.parameters(), 'lr': lr[0]},
                                        {'params': net.enc_out.parameters(), 'lr': lr[0]},
                                        {'params': net.classifier.parameters()}], lr=lr[1])
    else:
        optimizer = optim.NAdam(net.parameters(), lr=lr if isinstance(lr, float) else lr[0])

    time = dttm.datetime.now().strftime('%m/%d %H:%M:%S')
    msg = f'--> Started training {net_id} (time: {time})\n'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg)
    else:
        print(msg)

    saveat = os.path.join(res_dir,f'{net_id}.pth')
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
        if use_scheduler:
            if scheduler_patience:
                if isinstance(lr, float):
                    min_lr = lr*.01
                else:
                    min_lr = [lr[0]*.01 for _ in range(len(params)-1)] + [lr[1]*.01]        #gambiarra?
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
                    updater=optimizer, patience=patience, saveat=saveat, ae_train=ae_train,
                    start_from=start_from if start_from else None, device=device,
                    #start_from=saveat if (start_from or net_res) else None, device=device,
                    use_file=use_file, verbose=True, metrics=train_metrics, name=net_id,
                    scheduler=scheduler, min_lr=min_lr, ret_lr=ret_lr, minimize=minimize,
                    n_branches=n_branches, start_counting=start_counting,
                    nout_channels=nout_channels, ignore_class=ignore_class)#, l_when=l_criteria[0]+l_criteria[1]*i)

        #concatentate dict lists
        net_res = {key: value + aux[key] for key,value in net_res.items()} if net_res else aux
        del train_loader


    #save res --- good for plotting
    save_res = os.path.join(res_dir,f'{net_id}_tr.csv')
    DataFrame.from_dict(net_res).to_csv(save_res, index=False)

    res = defaultdict(list)
    res['net_id'].append(net_id)
    try:
        #testing config
        save_dict = tch.load(saveat)
        net.load_state_dict(save_dict['model_state_dict'])
        net.to(device)

        #evaluate model and save test results
        if n_branches:
            evaluator = eval_branches(n_branches)
        else:
            n_branches = 0
            evaluator = eval_results(count_one=True)#iter_mode=True,ae_mode=ae_train)    #talvez definir isso antes do train 
        for met,f in metrics:
            if met == 'mIoU':
                cur_res = f(net, n_branches + 1, nout_channels, ignore_class, test_loader, device)
                for keys, value in cur_res:
                    res[met].append(value)
            elif n_branches:
                cur_res = evaluator(net, test_loader, f, device)
                for key,value in cur_res.items():
                    res[f'{met}_{key}'].append(value)
                del cur_res
            else:
                res[met].append(evaluator(net, test_loader, f, device=device))
    except Exception as e:
        msg = f'The following error occurred while trying to evaluate network:\n{e}\n'
        if use_file:
            with open(use_file, 'a') as f:
                f.write(msg)
        else:
            print(msg)
        for key,_ in metrics:
            res[key].append(np.nan)

    time = dttm.datetime.now().strftime('%m/%d %H:%M:%S')
    msg = f'--> Finished training {net_id} (time: {time})\n'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg)
    else:
        print(msg)

    return res

#function to set up multiprocessing training
def eval_net(kwargs):
    og_dir = os.getcwd()
    res_dir = kwargs['res_dir']
    device = kwargs['device']

    model = kwargs['model']
    saveat = os.path.join(res_dir, model)
    kwargs['mod_dir'] = saveat
    save_res = os.path.join(saveat, model + '_res.csv')
    save_setup = os.path.join(saveat, model + '_su.csv')

    try:
        os.makedirs(saveat)
    except:
        pass

    val_set = kwargs['val_set']
    test_set = kwargs['test_set']

    with open(save_setup, 'w') as setup:
        setup.write(f'model: {model}\n')
        setup.write(f'act. func: {kwargs["act_func"]}\n')
        setup.write(f'conv. blk: {kwargs["conv_blk"]}\n')
        setup.write('net config:\n')
        setup.write('\n'.join([str(row) for row in kwargs['blk_config']]))
        setup.write('\n')
        if 'base_unet' in kwargs.keys():
            setup.write(f'base unet: {kwargs["base_unet"]}\n')

    #fixed loaders
    val_loader = tch.utils.data.DataLoader(val_set, batch_size=5, shuffle=False,
                                        num_workers=4, drop_last=False, prefetch_factor=4,
                                        pin_memory=True)
    test_loader = tch.utils.data.DataLoader(test_set, batch_size=5, shuffle=False,
                                        num_workers=4, drop_last=False, prefetch_factor=4,
                                        pin_memory=True)
    kwargs |= {'val_loader': val_loader, 'test_loader': test_loader}

    n_its = kwargs['n_rep'] if 'n_rep' in kwargs.keys() else 1

    if n_its > 1:
        net_ids = [model + f'_{i+1}' for i in range(n_its)]

        n_processes = kwargs['n_procs'] if 'n_procs' in kwargs.keys() else 1   #number of simultaneous processes
        with concurrent.ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = executor.map(_loop_iter,net_ids,repeat(kwargs))

        #get values and save results
        results = merge_dicts(list(results))
    else:
        results = _loop_iter(model,kwargs)
    DataFrame.from_dict(results).set_index('net_id').to_csv(save_res)

    return


#grid search loop
def _gs_loop_iter(params, kwargs):
    #n_classes = kwargs['n_classes']
    lr,batch_size = params
    blk_config = kwargs['blk_config'] if 'blk_config' in kwargs.keys() else None
    nin_channels = kwargs['nin_channels']
    nout_channels = kwargs['nout_channels']
    conv_blk = kwargs['conv_blk']
    conv_layer = get_conv[kwargs['conv_layer']] if 'conv_layer' in kwargs.keys() else nn.Conv2d
    dilation = kwargs['dilation'] if 'dilation' in kwargs.keys() else 1
    #classifier = kwargs['classifier']
    use_res = kwargs['use_res'] if 'use_res' in kwargs.keys() else False
    net_const = kwargs['net_const']
    val_set = kwargs['val_set']
    num_epochs = kwargs['num_epochs']
    #num_workers = kwargs['num_workers']
    act_func = kwargs['act_func']
    bias = kwargs['bias']
    device = kwargs['device'] if 'device' in kwargs.keys() else tch.device('cpu')
    initializer = get_initializer[kwargs['initializer']]

    #train config
    transform = kwargs['transforms'] if 'transforms' in kwargs.keys() else None
    ae_train = kwargs['ae_train']
    use_file = kwargs['use_file'] if 'use_file' in kwargs.keys() else None

    msg = f'Starting {kwargs["model"]} training training with lr = {lr} and b_size = {batch_size}'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg + '\n')
    else:
        print(msg)

    #prepare data loader
    num_workers = kwargs['def_nworkers'](batch_size)*tch.cuda.device_count()
    p_factor = kwargs['def_prefetch'](batch_size)
    val_loader = utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        drop_last=False, prefetch_factor=p_factor)
    #train config
    loss = get_loss[kwargs['loss']]
    add_parameters = dict()         #for unet-based classifiers
    for par in ['out_dim', 'exp_factor', 'base_net', 'n_classes', 'nencout_channels']:
        if par in kwargs.keys():
            if par == 'n_classes':
                add_parameters[par] = kwargs[par]+1
            else:
                add_parameters[par] = kwargs[par]

    net = net_const(nin_channels=nin_channels,
                    blk_config=blk_config,
                    nout_channels=nout_channels,
                    conv_block=conv_blk, dilation=dilation,
                    use_res=use_res,
                    conv_layer=conv_layer,
                    bias=bias,
                    act_func=act_func,
                    **add_parameters
                ).to(device)
    net.init_layers(initializer)

    #customize the lines below
    fixed_lr = kwargs['fixed_lr']
    if not ae_train and fixed_lr:
        optimizer = optim.NAdam([{'params': net.enc_block.parameters(), 'lr': fixed_lr},
                                {'params': net.enc_out.parameters(), 'lr': fixed_lr},
                                {'params': net.classifier.parameters()}], lr=lr)
    else:
        optimizer = optim.NAdam(net.parameters(), lr=lr)
    
    it_res = train(net, val_loader, loss, updater=optimizer, num_epochs=num_epochs,
                    device=device, verbose=False, ae_train=ae_train, use_file=use_file,
                    transform=transform, ret_lr=True)
    it_res |= {'batch_size': [batch_size for _ in it_res['lr']]}
    del net

    #queue.put(it_res)
    msg = f'Finished {kwargs["model"]} training training with lr = {lr} and b_size = {batch_size}'
    if use_file:
        with open(use_file, 'a') as f:
            f.write(msg + '\n')
    else:
        print(msg)

    return it_res

#function to set up multiprocessing training
def lr_search(kwargs : dict):
    model = kwargs['model']
    batch_sizes = kwargs['batch_sizes']
    lrs = kwargs['lrs']
    res_dir = kwargs['res_dir']

    saveat = os.path.join(res_dir, model)
    save_res = os.path.join(saveat, model + '.csv')

    try:
        os.makedirs(saveat)
    except:
        pass

    n_processes = kwargs['n_procs']             #number of simultaneous processes
    pairs = [(lr,b_size) for lr in lrs for b_size in batch_sizes]
    r_list = list()
    with concurrent.ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = executor.map(_gs_loop_iter,pairs,repeat(kwargs))

    #get values and save results
    results = merge_dicts(list(results))
    #modificar isso no programa principal
    DataFrame.from_dict(results).set_index('lr').to_csv(
            save_res)

    return saveat
