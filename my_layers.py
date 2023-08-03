#! /usr/bin/python3

from common_header import *
from common_torch import *

class Lambda_layer(nn.Module):
    #from https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903
    #changed the names to my style
    def __init__(self,function) -> None:
        super(Lambda_layer,self).__init__()
        self.f = function

    def forward(self,inputs : Tensor) -> Tensor:
        return self.f(inputs)

class DenseLayer(nn.Sequential):
    def __init__(self, inp_dim : int, out_dim : int,
            norm_layer : Optional[Callable[...,nn.Module]] =None,
            act_func : str ='relu', **kwargs) -> None:

        if act_func == 'relu6':
            activation_layer = nn.ReLU6
        elif act_func == 'selu':
            activation_layer = nn.SELU
        elif act_func == 'leaky relu':
            activation_layer = nn.LeakyReLU
        elif act_func == 'elu':
            activation_layer = nn.ELU
        elif act_func == 'celu':
            activation_layer = nn.CELU
        elif act_func == 'mish':
            activation_layer = nn.Mish
        else:
            activation_layer = nn.ReLU

        config = [nn.Linear(inp_dim, out_dim)]

        if norm_layer is None and re.search(
                'relu',act_func,flags=re.IGNORECASE):
            config.append(nn.BatchNorm1d(1))
        config.append(activation_layer(inplace=True))

        super(DenseLayer,self).__init__(*config)

class ConvLayer(nn.Sequential): #adapted from pytorch site
    def __init__(self, nin_channels : int, nout_channels : int,
            n_convs : int =1, kernel_size : int =3, stride : int =1,
            dilation : int =1, n_groups : int =1,
            conv_layer : Optional[Callable[...,nn.Module]] =None,
            norm_layer : Optional[Callable[...,nn.Module]] =None,
            act_func : str ='relu', #use_res : bool =False,
            bias : bool =False, **kwargs) -> None:

        padding = (kernel_size - 1) // 2*dilation
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None and re.search(
                'relu',act_func,flags=re.IGNORECASE):
            norm_layer = nn.BatchNorm2d

        if act_func == 'relu6':
            activation_layer = nn.ReLU6
        elif act_func == 'selu':
            activation_layer = nn.SELU
        elif act_func == 'leaky relu':
            activation_layer = nn.LeakyReLU
        elif act_func == 'elu':
            activation_layer = nn.ELU
        elif act_func == 'celu':
            activation_layer = nn.CELU
        elif act_func == 'mish':
            activation_layer = nn.Mish
        elif act_func == 'relu':
            activation_layer = nn.ReLU
        else:
            activation_layer = None

   #     self.use_res_connect = use_res and \
   #         nin_channels == nout_channels and stride == 1

        config = [
                    conv_layer(nin_channels, nout_channels,
                            kernel_size, stride, padding,
                            dilation=dilation, groups=n_groups,
                            bias=bias) #for i in range(n_convs)
                    if conv_layer in [nn.Conv1d, nn.Conv2d, nn.Conv3d] else
                    conv_layer(nin_channels, nout_channels,
                            kernel_size, stride, padding,
                            output_padding=padding,
                            dilation=dilation, groups=n_groups,
                            bias=bias) for _ in range(n_convs)

                    ]
        if norm_layer: 
            config.append(norm_layer(nout_channels))
        if activation_layer:
            config.append(activation_layer(inplace=True))

        super(ConvLayer,self).__init__(*config)

#class Rearrange(nn.Module):
#    def __init__(self, n : int) -> None:
#        super(Rearrange,self).__init__()
#        self.n = n
#
#    def forward(self,x : Tensor) -> Tensor:
#        if x.dim() < 4:
#            return cat([x[:, i::self.n, :] for i in range(self.n)], dim=1)
#        return cat([x[:, i::self.n, :, :] for i in range(self.n)], dim=1)

class Base_layer(nn.Module):
    def __init__(self,depth : int, weights : Tensor,
            conv_layer : Optional[Callable[...,nn.Module]] =None) -> None:

        super().__init__()
        if conv_layer is None:
            conv_layer = nn.ConvTranspose1d

        self.depth = depth
        aux_base = list()
        self.scal_factor = [1] + [
                1/(sqrt(2)**i) for i in range(depth)]
        n = weights.shape.numel()

        aux_base.append(
                conv_layer(1, 1, n, stride=n, bias=False)
            )
        aux_base[0].weight = nn.Parameter(
                tensor(
                    [1. for _ in range(weights.shape.numel())]
                ).reshape(1,1,-1))
        for i in range(1,self.depth+1):
            aux_base.append(
                conv_layer(1, 1, n, stride=n, bias=False)
            )
            aux_base[i].weight = weights
        self.base = nn.ModuleList(aux_base)

        split_domains = lambda x: [x[0]] + [
                x[max(2**i,1):2**(i+1)] for i in range(self.depth)]
        self.split_input = Lambda_layer(split_domains)
        self.network = nn.Sequential(self.split_input, self.base)

    def forward(self,inputs : Tensor) -> Tensor:
        assert 2**self.depth == inputs.shape.numel(), f'Inconpatible shape!!! Input must consist of {2**self.depth} constants!'

        inputs = self.split_input(inputs)
        int_results = m_process.Pool.map(lambda x,w,a: a*w(x.reshape(1,1,-1)),
            inputs, self.base, self.scal_factor)
        n = int_results[-1].size(dim=2)
        decomp = list()
        for res in int_results:
            l = res.size(dim=2)
            decomp.append(nn.functional.interpolate(
                res,scale_factor=(n//l),mode='linear')
                if l != n else res)
        return stack(decomp, dim=0).sum(dim=0)

class DWT_layer(nn.Module):
    def __init__(self,dwt='haar',pad=False):
        super(DWT_layer,self).__init__()
        self.pad = pad
        self.requires_grad = False
        self.dwt = dwt              #this is a dummy, for later use

    def _haar_dwt(self,x):
        '''
            Taken from Multi-Level Wavelet Convolutional Neural Networks
            Liu et. al.
            I've changed x_i (i = 1,2,3,4) to indicate even/odd,
            because I think it is more didatic. Added the original
            nomenclature for comparison.
        '''
        x_e = x[:,:,::2,:]#/2
        x_o = x[:,:,1::2,:]#/2
        x_ee = x_e[:,:,:,::2]               #x_1
        x_eo = x_e[:,:,:,1::2]              #x_3
        x_oe = x_o[:,:,:,::2]               #x_2
        x_oo = x_o[:,:,:,1::2]              #x_4
        x_ll = x_ee + x_oe + x_eo + x_oo
        x_hl = -x_ee - x_oe + x_eo + x_oo
        x_lh = -x_ee + x_oe - x_eo + x_oo
        x_hh = x_ee - x_oe - x_eo + x_oo

        return cat((x_ll,x_lh,x_hl,x_hh), 1)

    def forward(self,x):
        if self.pad:
            if (x.shape[-1] // 2) % 2:
                x = F.pad(x,(1,1), 'constant', 0)
            if (x.shape[-2] // 2) % 2:
                x = F.pad(x,(0,0,1,1), 'constant', 0)
        return self._haar_dwt(x)

#add IWT layer

#add multi resolution conv_blk

def get_layers(block : nn.Module) -> List:
    # addapted from https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    children = list(block.children())
    layers = list()
    if children == []:
        return children
    
    for child in children:
        if isinstance(child, nn.Module):
            layers.extend(get_layers(child))
        else:
            layers.append(child)

    return layers
