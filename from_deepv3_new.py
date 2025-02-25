#! /usr/bin/python3

from common_header import *
from my_layers import ConvLayer, get_layers
from allocate_cuda_device import allocate_cuda
import torch as tch
import torchvision
from pthflops import count_ops
from torch import nn, cat, load
import copy
from module_variables import *
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP

class my_branch(nn.Sequential):
    def __init__(self, nin_channels,
                       num_classes,
                       atrous_rates,
                       nout_channels,
                       bottleneck=None,
                       **kwargs,
                ):
        if bottleneck:
            super().__init__(
                nn.Conv2d(nin_channels, bottleneck, 1),
                ASPP(bottleneck, atrous_rates, nout_channels),
                nn.Conv2d(nout_channels, nout_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(nout_channels),
                nn.ReLU(),
                nn.Conv2d(nout_channels, num_classes, 1),
            )
        else:
            super().__init__(
                ASPP(nin_channels, atrous_rates, nout_channels),
                nn.Conv2d(nout_channels, nout_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(nout_channels),
                nn.ReLU(),
                nn.Conv2d(nout_channels, num_classes, 1),
            )

def get_base_model(name, model='deeplabv3_resnet101', pretrained=True):
    try:
        trained_model = load(name)
    except:
        if re.search('deeplabv3', model):
            if re.search('resnet50', model):
                trained_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)#True)
            else:
                trained_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)#True)
        else:
            pass
    if not os.path.exists(name):
        tch.save(trained_model, name)
    return trained_model

class branchyDeepv3(nn.Module):
    def __init__(self, base_name, base_type, n, img_dim, count_branches=True, skip=0, branch_params=None):
        super(branchyDeepv3,self).__init__()
        aux_model = get_base_model(base_name, base_type)
        self.classifier = copy.deepcopy(aux_model.classifier)
        self.count_branches = count_branches
        device = allocate_cuda()

        #initialization
        self.base_model = list()
        self.n_branches = n
        self.branches = list()
        tot_flops = self.__check_flops(aux_model.backbone, img_dim, device)
        flop_pos = tot_flops/(self.n_branches + 1)

        #insert branches
        aux_base = list()
        section = list()
        input_layers = True
        for layer in list(aux_model.backbone.named_modules())[1:]:
            n_branches = len(self.branches)
            if input_layers and not re.match(r'layer', layer[0]):
                aux_base.append(copy.deepcopy(layer[1]))
                section.append(copy.deepcopy(layer[1]))
            elif isinstance(layer[0],str) and re.match(r'layer[0-9]+.[0-9]+$', layer[0]):                 #named features that we extract from base model
                aux_base.append(layer[1])
                section.append(layer[1])
                if (n > n_branches) and tot_flops > self.__check_flops(aux_base,img_dim, device) > flop_pos*(n_branches + (1+skip)):
                    self.base_model.append(nn.Sequential(*section))
                    nin_channels = self.__check_nout_channels(aux_base, device)
                    self.branches.append(self.__gen_branch(nin_channels, nout_channels=21, branch_params=branch_params))
                    section = list()
            else:
                input_layers = False
        self.base_model.append(nn.Sequential(*section))
        del aux_base, section

        self.base_model = nn.ModuleList(self.base_model)
        self.branches = nn.ModuleList(self.branches)
        self.n_branches = len(self.branches)
        del aux_model
        self.__init_branches()

    def __check_flops(self, aux_model, img_dim, device):
        if isinstance(aux_model, list):
            aux = aux_model
            aux_model = nn.Sequential(*aux_model)
        aux_model.to(device)
        aux_model.eval()
        tensor = tch.rand(1,3,img_dim,img_dim).to(device)
        tot_flops,_ = count_ops(aux_model, tensor, print_readable=False, verbose=False)

        if len(self.branches) and self.count_branches:
            for i,branch in enumerate(self.branches):
                branch.to(device)
                tensor = self.base_model[i](tensor)
                b_flops,_ =  count_ops(branch, tensor, print_readable=False, verbose=False)
                tot_flops += b_flops
        del aux_model
        return tot_flops

    def __check_nout_channels(self, aux_model, device):
        aux_model = nn.Sequential(*aux_model)
        aux_model.to(device)
        aux_model.eval()
        tensor = tch.rand(1, 3, 100, 100).to(device)
        nout_channels = aux_model(tensor).shape[1]
        del aux_model
        return nout_channels

    def __gen_branch(self, nin_channels, nout_channels=21, branch_params=None):
        if isinstance(branch_params, dict) and all(k in branch_params for k in ('nout_channels', 'atrous_rates')):
            return my_branch(nin_channels=nin_channels, num_classes=nout_channels, **branch_params)

        #default
        return DeepLabHead(nin_channels, nout_channels)

    def __init_branches(self):
        for branch in self.branches:
            for layer in get_layers(branch):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)


    def forward(self, X):
        outputs = list()
        inp_shape = X.shape[-2:]
        for i in range(self.n_branches):
            X = self.base_model[i](X)
            br_output = self.branches[i](X)
            br_output = F.interpolate(br_output, size=inp_shape, mode='bilinear', align_corners=False)
            outputs.append(br_output.unsqueeze(0))
        y = self.classifier(self.base_model[-1](X))
        out = F.interpolate(y, size=inp_shape, mode='bilinear', align_corners=False)
        outputs.append(out.unsqueeze(0))

        return cat(outputs)

if __name__ == '__main__':
    from torchinfo import summary
    import inspect

    model = branchyDeepv3('base_model.pth', 'deeplabv3_resnet101', 10, 256)
    model.eval()
    summary(model, (1, 3, 256, 256))
