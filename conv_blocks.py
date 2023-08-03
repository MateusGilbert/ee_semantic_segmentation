#! /usr/bin/python3

from common_torch import *
from my_layers import *
from aux_functions import next_div

def get_actfunction(act_func: str) -> Callable:
    if act_func == 'relu6':
        return nn.ReLU6
    if act_func == 'selu':
        return nn.SELU
    if act_func == 'leaky relu':
        return nn.LeakyReLU
    if act_func == 'elu':
        return nn.ELU
    if act_func == 'celu':
        return nn.CELU
    if act_func == 'mish':
        return nn.Mish
    return nn.ReLU

class InvertedResidual(nn.Module):   #from pytorch site
    def __init__(self, nin_channels : int, nout_channels : int,
            expand_ratio : int =1, kernel_size : int =3,
            stride : int =1, dilation : int =1, n_groups : int =1,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            act_func : str ='relu6', use_res : bool =False,
            **kwargs) -> None:

        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if conv_layer is None:
            conv_layer = nn.Conv2d

        if norm_layer is None and \
            re.search('relu', act_func, flags=re.IGNORECASE):
            if conv_layer == nn.Conv1d:
                norm_layer = nn.BatchNorm1d
            elif conv_layer == nn.Conv3d:
                norm_layer = nn.BatchNorm3d
            else:
                norm_layer = nn.BatchNorm2d

        nhidden_channels = int(round(nin_channels * expand_ratio))
        self.use_res = use_res and \
            nin_channels == nout_channels and stride == 1

        layers = list()
        if expand_ratio != 1:
            # pw
            layers.append(
                    ConvLayer(nin_channels, nhidden_channels,
                        kernel_size=1, norm_layer=norm_layer,
                        act_func=act_func, conv_layer=conv_layer)
                    )
        layers.extend([
            # dw
            ConvLayer(nhidden_channels, nhidden_channels,
                stride=stride, n_groups=nhidden_channels,
                norm_layer=norm_layer, act_func=act_func,
                conv_layer=conv_layer),
            # pw-linear
            conv_layer(nhidden_channels, nout_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
        ])
        if norm_layer:
            layers.append(norm_layer(nout_channels))

        self.conv = nn.Sequential(*layers)
        self.act = get_actfunction(act_func)(inplace=True)

    def _forward_imp(self, x: Tensor) -> Tensor:
        if self.use_res:
            return self.act( x + self.conv(x) )
        else:
            return self.act(self.conv(x))

    def forward(self, x : Tensor) -> Tensor:
        return self._forward_imp(x)

class InceptionBlk(nn.Module):
    def __init__(self, nin_channels : int, nout_1x1 : int,
            nout_3x3 : int, nout_5x5 : int, nout_pool : int,
            nred_3x3 : int, nred_5x5 : int,
            dilation : int =1, n_groups : int =1,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            pool_layer : Optional[Callable[..., nn.Module]] =None,
            act_func : str ='relu6', use_res : bool =False,
            bias : bool =True, **kwargs) -> None:

        super(InceptionBlk, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d

        if norm_layer is None and \
            re.search('relu', act_func, flags=re.IGNORECASE):
            if conv_layer == nn.Conv1d:
                norm_layer = nn.BatchNorm1d
            elif conv_layer == nn.Conv3d:
                norm_layer = nn.BatchNorm3d
            else:
                norm_layer = nn.BatchNorm2d

        if pool_layer is None:
            if conv_layer == nn.Conv1d:
                pool_layer = nn.MaxPool1d
            elif conv_layer == nn.Conv3d:
                pool_layer = nn.MaxPool3d
            else:
                pool_layer = nn.MaxPool2d

        self.use_res= use_res and \
            nin_channels == (nout_1x1 + nout_3x3 + nout_5x5 + nout_pool)

        self.layer_1x1 = ConvLayer(nin_channels, nout_1x1,
                            kernel_size=1, norm_layer=norm_layer,
                            act_func=act_func, conv_layer=conv_layer,
                            bias=bias)
        self.layer_3x3 = nn.Sequential(
                            ConvLayer(nin_channels, nred_3x3,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                            ConvLayer(nred_3x3, nout_3x3,
                                kernel_size=3, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                        )
        self.layer_5x5 = nn.Sequential(
                            ConvLayer(nin_channels, nred_5x5,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                            ConvLayer(nred_5x5, nout_5x5,
                                kernel_size=5, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                        )
        self.layer_pool = nn.Sequential(
                            pool_layer(kernel_size=3,padding=1,stride=1),
                            ConvLayer(nin_channels, nout_pool,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                )

    def forward(self,x : Tensor) -> Tensor:
        outputs = [
                    self.layer_1x1(x),
                    self.layer_3x3(x),
                    self.layer_5x5(x),
                    self.layer_pool(x)
                ]

        if self.use_res:
            return cat(outputs,dim=1) + x
        return cat(outputs,dim=1)

#customizable inception block
class CIncepBlk(nn.Module):
    def __init__(self, nin_channels : int, nout_1x1 : int,
            nout_3x3 : int, nout_5x5 : int, nout_pool : int,
            nred_3x3 : int, nred_5x5 : int,
            dilation : int =1, conv_block : nn.Module =None,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            pool_layer : Optional[Callable[..., nn.Module]] =None,
            act_func : str ='relu6', use_res : bool =False,
            bias : bool =True, **kwargs) -> None:

        super(CIncepBlk, self).__init__()

        if conv_layer is None:
            conv_layer = nn.Conv2d

        if conv_block is None:
            conv_block = InceptionBlk

        if norm_layer is None and \
            re.search('relu', act_func, flags=re.IGNORECASE):
            if conv_layer == nn.Conv1d:
                norm_layer = nn.BatchNorm1d
            elif conv_layer == nn.Conv3d:
                norm_layer = nn.BatchNorm3d
            else:
                norm_layer = nn.BatchNorm2d

        if pool_layer is None:
            if conv_layer == nn.Conv1d:
                pool_layer = nn.MaxPool1d
            elif conv_layer == nn.Conv3d:
                pool_layer = nn.MaxPool3d
            else:
                pool_layer = nn.MaxPool2d

        self.use_res= use_res and \
            nin_channels == (nout_1x1 + nout_3x3 + nout_5x5 + nout_pool)

        self.layer_1x1 = ConvLayer(nin_channels, nout_1x1,
                            kernel_size=1, norm_layer=norm_layer,
                            act_func=act_func, conv_layer=conv_layer,
                            bias=bias)
        self.layer_3x3 = nn.Sequential(
                            ConvLayer(nin_channels, nred_3x3,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                            conv_block(nred_3x3, nout_3x3,kernel_size=3,
                                norm_layer=norm_layer, act_func=act_func,
                                conv_layer=conv_layer, bias=bias,
                                dilation=dilation),
                        )
        self.layer_5x5 = nn.Sequential(
                            ConvLayer(nin_channels, nred_5x5,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                            conv_block(nred_5x5, nout_5x5, kernel_size=5,
                                norm_layer=norm_layer, act_func=act_func,
                                conv_layer=conv_layer, bias=bias,
                                dilation=dilation),
                        )
        self.layer_pool = nn.Sequential(
                            pool_layer(kernel_size=3,padding=1,stride=1),
                            ConvLayer(nin_channels, nout_pool,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias),
                )

    def forward(self,x : Tensor) -> Tensor:
        outputs = [
                    self.layer_1x1(x),
                    self.layer_3x3(x),
                    self.layer_5x5(x),
                    self.layer_pool(x)
                ]

        if self.use_res:
            return cat(outputs,dim=1) + x
        return cat(outputs,dim=1)

#Hybrid Dilated Convolution Block
class HDConvBlk(nn.Module):
    def __init__(self, nin_channels : int, nout_channels : int,
            kernel_size : int =3, stride : int =1, dilation : int =1,
            conv_layer: Optional[Callable[..., nn.Module]] =None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            act_func : str ='relu6', use_res : bool =False,
            combine_channels : bool =True, rearrange : bool=True,
            **kwargs) -> None:

        super(HDConvBlk, self).__init__()

        padding = lambda d: (kernel_size - 1) // 2*d

        if conv_layer is None:
            conv_layer = nn.Conv2d

        if norm_layer is None and \
            re.search('relu',act_func,flags=re.IGNORECASE):
            if conv_layer == nn.Conv1d:
                norm_layer = nn.BatchNorm1d
            elif conv_layer == nn.Conv3d:
                norm_layer = nn.BatchNorm3d
            else:
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
        else:
            activation_layer = nn.ReLU

        self.use_res= use_res and \
            nin_channels == nout_channels and stride == 1

        self.n_sections = dilation      #number of dilations
    
        if nin_channels > nout_channels or \
           nout_channels % nin_channels != 0:
            if nin_channels > nout_channels:
                norm_channels = nout_channels
            else:
                norm_channels = next_div(nin_channels,
                                        nout_channels)
            self.norm_channels = conv_layer(
                                                nin_channels,
                                                norm_channels,
                                                kernel_size=1,
                                            )
        else:
            self.norm_channels = None
            norm_channels = nin_channels

        n_groups = min(norm_channels,nout_channels)

        self.dil_blks = nn.ModuleList([
                conv_layer(norm_channels, nout_channels, kernel_size,
                        stride, padding(i), dilation=i, groups=n_groups,
                        bias=False) for i in range(1,dilation+1)
            ])

        self.batch_norm = norm_layer(nout_channels*dilation) if norm_layer else None
#           self.act_layer = nn.Sequential(
#                   norm_layer(nout_channels*dilation),
#                   activation_layer(inplace=True))
#       else:
        self.act_layer = activation_layer(inplace=True)

        self.rearrange = rearrange
        self.n = dilation
        self.comb_ch = combine_channels
        if combine_channels:
            self.out_layer = ConvLayer(dilation*nout_channels, nout_channels,
                                    kernel_size=1, n_groups=nout_channels,#dilation,
                                    conv_layer=conv_layer, norm_layer=norm_layer,
                                    act_func=None#', use_res=use_res
                               )

    def forward(self,x : Tensor) -> Tensor:
        if self.norm_channels != None:
            x = self.norm_channels(x)

        outputs = [
                self.dil_blks[i](x)
                for i in range(self.n_sections)
            ]

        if self.batch_norm:
            outputs = self.batch_norm(self.act_layer(cat(outputs,dim=1)))
        else:
            outputs = self.act_layer(cat(outputs,dim=1))


        if self.rearrange:
            outputs = cat([outputs[:, i::self.n] for i in range(self.n)], dim=1)

        if self.comb_ch:
            outputs = self.out_layer(outputs)

        if self.use_res:
            return self.act_layer(outputs + x)
        return self.act_layer(outputs)

class Haar_transform(nn.Module):
    def __init__(self, inverse : bool =False):
        super(Haar_transform, self).__init__()
        self.requieres_grad = False
        self.inverse = inverse

    def _transf(self, x: Tensor) -> Tensor:
        x_e = x[:, :, ::2, :]
        x_o = x[:, :, 1::2, :]
        x_ee = x_e[:, :, :, ::2]
        x_eo = x_e[:, :, :, 1::2]
        x_oe = x_o[:, :, :, ::2]
        x_oo = x_o[:, :, :, 1::2]

        x_ll = x_ee + x_oe + x_eo + x_oo
        x_hl = -x_ee - x_oe + x_eo + x_oo
        x_lh = -x_ee + x_oe - x_eo + x_oo
        x_hh = x_ee - x_oe - x_eo + x_oo

        return cat((x_ll, x_hl, x_lh, x_hh), 1)

    def _inv_transf(self, x: Tensor) -> Tensor:
        N, C, h, w  = x.size()
        C_out = C // 4
        h_out = h*2
        w_out = w*2
        x_ll = x[:, :C_out, :, :]
        x_hl = x[:, C_out:2*C_out, :, :]
        x_lh = x[:, 2*C_out:3*C_out, :, :]
        x_hh = x[:, 3*C_out:, :, :]

        x_out = zeros([N,C_out,h_out,w_out]).float().to(device=x.device)

        x_out[:,:,::2,::2]   = (x_ll - x_hl - x_lh + x_hh) / 4
        x_out[:,:,::2,1::2]  = (x_ll + x_hl - x_lh - x_hh) / 4
        x_out[:,:,1::2,::2]  = (x_ll - x_hl + x_lh - x_hh) / 4
        x_out[:,:,1::2,1::2] = (x_ll + x_hl + x_lh + x_hh) / 4

        return x_out

    def forward(self, x : Tensor) -> Tensor:
        if self.inverse:
            return self._inv_transf(x)
        return self._transf(x)

class WaveBlk(InvertedResidual):
    def __init__(self, nin_channels : int, nout_channels : int,
            kernel_size : int =3, stride : int =1, dilation : int =1,
            conv_layer: Optional[Callable[..., nn.Module]] =None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            act_func : str ='relu6', use_res : bool =False,
            combine_channels : bool =True, rearrange : bool=True,
            **kwargs) -> None:

        assert stride in [1, 2, .5]

        if stride > 1:
            nin_channels *= 4
        elif stride < 1:
            nin_channels /= 4

        super(WaveBlk, self).__init__(
                                        nin_channels=nin_channels,
                                        nout_channels=nout_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        dilation=dilation,
                                        conv_layer=conv_layer,
                                        norm_layer=norm_layer,
                                        act_func=act_func,
                                        use_res=use_res,
                                        combine_channels=combine_channels,
                                        rearrange=rearrange,
                                        **kwargs
                                    )
        if stride > 1:
            self.transf = Haar_transform()
        elif stride < 1:
            self.transf = Haar_transform(inverse=True)
        else:
            self.transf = None

    def forward(self, x : Tensor) -> Tensor:
        if self.transf:
            x = self.transf(x)
        return self._forward_imp(x)

#create a block with dilation and no padding, using upsampling

#U-net blocks
class DownBlk(nn.Module):
    def __init__(self, nin_channels : int, nout_channels : int,
            nred_channels : int =None, n_convs : int=1,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            pool_layer : Optional[Callable[..., nn.Module]] = None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            conv_block : nn.Module =None, act_func : str ='relu6',
            use_res : bool =False, bias : bool=False, **kwargs) -> None:
        super(DownBlk,self).__init__()

        if conv_block is None:
            conv_block = ConvLayer

        if conv_layer is None:
            conv_layer = nn.Conv2d

        self.conv = conv_block(nin_channels, nout_channels,
                        norm_layer=norm_layer, act_func=act_func,
                        conv_layer=conv_layer, use_res=use_res, bias=bias,
                        n_convs=n_convs, **kwargs
                    )

        if pool_layer:
            self.down = pool_layer(kernel_size=3, padding=1, stride=2)
        else:
            self.down = conv_block(nout_channels, nout_channels, stride=2,
                        norm_layer=norm_layer, act_func=act_func,
                        conv_layer=conv_layer, use_res=use_res, bias=False,
                        n_groups=nout_channels
                    )

    def forward(self, x : Tensor) -> Tuple[Tensor,Tensor]:
        pre_down = self.conv(x)
        return self.down(pre_down), pre_down

class UpBlk(nn.Module):
    def __init__(self, nin_channels : int, nout_channels : int,
            nred_channels : int =None, n_convs : int=1,
            nbypass_channels : int=None,
            bilinear : bool=True, crop : bool=True,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            pool_layer : Optional[Callable[..., nn.Module]] = None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            conv_block : nn.Module =None, act_func : str ='relu6',
            use_res : bool =False, bias : bool=False, **kwargs) -> None:
        super(UpBlk,self).__init__()

        self.crop = crop

        if conv_block is None:
            conv_block = ConvLayer

        if conv_layer is None:
            conv_layer = nn.Conv2d

        #nred_channels = nred_channels or (nin_channels // 2)
        nred_channels = nred_channels or nout_channels      #check

        if bilinear:
            self.crop = False       ################
            #adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
            up = [
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    #change from the original u-net
                    ConvLayer(nin_channels, nred_channels,
                        kernel_size=1, norm_layer=norm_layer,
                        act_func=act_func, conv_layer=conv_layer,
                        bias=bias)
                ]
        else:
            if pool_layer:
                up = [
                        pool_layer(kernel_size=3, padding=1, stride=2),
                        #change from the original u-net
                        ConvLayer(nin_channels, nred_channels,
                            kernel_size=1, norm_layer=norm_layer,
                            act_func=act_func, conv_layer=conv_layer,
                            bias=bias)
                    ]
            else:
                if conv_layer == nn.Conv1d:
                    t_conv = nn.ConvTranspose1d
                elif conv_layer == nn.Conv3d:
                    t_conv = nn.ConvTranspose3d
                else:
                    t_conv = nn.ConvTranspose2d
                up = ConvLayer(nin_channels, nred_channels,
                            stride=2, norm_layer=norm_layer,
                            act_func=act_func, conv_layer=t_conv,
                            n_groups=nred_channels, bias=bias)     #instead of max-pooling

        if nbypass_channels:
            nin_channels = nred_channels + nbypass_channels

        conv = [
                conv_block(nin_channels, nred_channels,
                    norm_layer=norm_layer, act_func=act_func,
                    conv_layer=conv_layer, use_res=use_res, bias=bias,
                    **kwargs)
            ]
        conv.extend([
                conv_block(nred_channels, nred_channels,
                    norm_layer=norm_layer, act_func=act_func,
                    conv_layer=conv_layer, use_res=use_res, bias=bias,
                    **kwargs
                )
                for _ in range(n_convs - 2)
            ])
        conv.append(conv_block(nred_channels, nout_channels,
                    norm_layer=norm_layer, act_func=act_func,
                    conv_layer=conv_layer, use_res=use_res, bias=bias,
                    **kwargs))

        self.conv = nn.Sequential(*conv)
        self.up = nn.Sequential(*up) if isinstance(up,list) else up

    def forward(self, x1 : Tensor, x2 : Tensor) -> Tensor:
        x1 = self.up(x1)

        if x1.shape[2:] != x2.shape[2:]:
            x_diff = x2.size()[2] - x1.size()[2]
            if x2.dim() > 3:
                y_diff = x2.size()[3] - x1.size()[3]

            if self.crop:
                if y_diff in locals():
                    x2 = x2[:,:,
                            x_diff//2:x2.size()[2] - x_diff//2,y_diff//2:x2.size()[2] - y_diff//2]
                else:
                    x2 = x2[:,:,x_diff//2:x2.size()[2] - x_diff//2]
            else:
                if y_diff in locals():#############conferir
                    x1 = F.pad(x1, [x_diff//2, x_diff - x_diff//2, y_diff//2, y_diff - y_diff//2])
                else:
                    x1 = F.pad(x1, [x_diff//2, x_diff - x_diff//2])

        return self.conv(cat([x2, x1], dim=1))

#class UpBlk(nn.Module):
#   def __init__(self, nin_channels : int, nout_channels : int,
#           nred_channels : int =None, n_convs : int=1,
#           bilinear : bool=True, crop : bool=True,
#           conv_layer : Optional[Callable[..., nn.Module]] =None,
#           pool_layer : Optional[Callable[..., nn.Module]] = None,
#           norm_layer : Optional[Callable[..., nn.Module]] =None,
#           conv_block : nn.Module =None, act_func : str ='relu6',
#           use_res : bool =False, bias : bool=False, **kwargs) -> None:
#       super(UpBlk,self).__init__()

#       self.crop = crop

#       if conv_block is None:
#           conv_block = ConvLayer

#       if conv_layer is None:
#           conv_layer = nn.Conv2d

#       #nred_channels = nred_channels or (nin_channels // 2)
#       nred_channels = nred_channels or nout_channels      #check

#       if bilinear:
#           self.crop = False       ################
#           #adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
#           up = [
#                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                   #change from the original u-net
#                   ConvLayer(nin_channels, nred_channels,
#                       kernel_size=1, norm_layer=norm_layer,
#                       act_func=act_func, conv_layer=conv_layer,
#                       bias=bias)
#               ]
#       else:
#           if pool_layer:
#               up = [
#                       pool_layer(kernel_size=3, padding=1, stride=2),
#                       #change from the original u-net
#                       ConvLayer(nin_channels, nred_channels,
#                           kernel_size=1, norm_layer=norm_layer,
#                           act_func=act_func, conv_layer=conv_layer,
#                           bias=bias)
#                   ]
#           else:
#               if conv_layer == nn.Conv1d:
#                   t_conv = nn.ConvTranspose1d
#               elif conv_layer == nn.Conv3d:
#                   t_conv = nn.ConvTranspose3d
#               else:
#                   t_conv = nn.ConvTranspose2d
#               up = ConvLayer(nin_channels, nred_channels,
#                           stride=2, norm_layer=norm_layer,
#                           act_func=act_func, conv_layer=t_conv,
#                           n_groups=nred_channels, bias=bias)     #instead of max-pooling

#       if conv_block == HDConvBlk or ('add_red' in kwargs.keys() and kwargs['add_red']):#gambiarra
#           nin_channels = nout_channels + nred_channels

#       conv = [
#               conv_block(nin_channels, nred_channels,
#                   norm_layer=norm_layer, act_func=act_func,
#                   conv_layer=conv_layer, use_res=use_res, bias=bias,
#                   **kwargs)
#           ]
#       conv.extend([
#               conv_block(nred_channels, nred_channels,
#                   norm_layer=norm_layer, act_func=act_func,
#                   conv_layer=conv_layer, use_res=use_res, bias=bias,
#                   **kwargs
#               )
#               for _ in range(n_convs - 2)
#           ])
#       conv.append(conv_block(nred_channels, nout_channels,
#                   norm_layer=norm_layer, act_func=act_func,
#                   conv_layer=conv_layer, use_res=use_res, bias=bias,
#                   **kwargs))

#       self.conv = nn.Sequential(*conv)
#       self.up = nn.Sequential(*up) if isinstance(up,list) else up

#   def forward(self, x1 : Tensor, x2 : Tensor) -> Tensor:
#       x1 = self.up(x1)

#       if x1.shape[2:] != x2.shape[2:]:
#           x_diff = x2.size()[2] - x1.size()[2]
#           if x2.dim() > 3:
#               y_diff = x2.size()[3] - x1.size()[3]

#           if self.crop:
#               if y_diff in locals():
#                   x2 = x2[:,:,
#                           x_diff//2:x2.size()[2] - x_diff//2,y_diff//2:x2.size()[2] - y_diff//2]
#               else:
#                   x2 = x2[:,:,x_diff//2:x2.size()[2] - x_diff//2]
#           else:
#               if y_diff in locals():#############conferir
#                   x1 = F.pad(x1, [x_diff//2, x_diff - x_diff//2, y_diff//2, y_diff - y_diff//2])
#               else:
#                   x1 = F.pad(x1, [x_diff//2, x_diff - x_diff//2])

#       return self.conv(cat([x2, x1], dim=1))

class Seg_Branch(nn.Module):
    def __init__(self, nin_channels : List[int],
            n_convs : int=1, bilinear : bool=True, crop : bool=True,
            conv_layer : Optional[Callable[..., nn.Module]] =None,
            pool_layer : Optional[Callable[..., nn.Module]] = None,
            norm_layer : Optional[Callable[..., nn.Module]] =None,
            conv_block : nn.Module =None, act_func : str ='relu6',
            use_res : bool =False, bias : bool=False, **kwargs) -> None:
        super(Seg_Branch,self).__init__()

        self.crop = crop
        self.n_inputs = len(nin_channels)

        if conv_block is None:
            conv_block = ConvLayer

        if conv_layer is None:
            conv_layer = nn.Conv2d

        self.intesections = list()
        x_channels = nin_channels[0]
        ups = list()
        convs = list()
        for pre_channels in nin_channels[1:]:
            nred_channels = pre_channels // 2 if pre_channels > nin_channels[-1] else pre_channels
            nout_channels = pre_channels

            if bilinear:
                self.crop = False
                #adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
                ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        #change from the original u-net
                        ConvLayer(x_channels, nred_channels,
                            kernel_size=1, norm_layer=norm_layer,
                            act_func=act_func, conv_layer=conv_layer,
                            bias=bias)
                    )
                )
            else:
                if pool_layer:
                    ups.append(
                        nn.Sequential(
                            pool_layer(kernel_size=3, padding=1, stride=2),
                            #change from the original u-net
                            ConvLayer(x_channels, nred_channels,
                                kernel_size=1, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=conv_layer,
                                bias=bias)
                        )
                    )
                else:
                    if conv_layer == nn.Conv1d:
                        t_conv = nn.ConvTranspose1d
                    elif conv_layer == nn.Conv3d:
                        t_conv = nn.ConvTranspose3d
                    else:
                        t_conv = nn.ConvTranspose2d
                    ups.append(
                            ConvLayer(x_channels, nred_channels,
                                stride=2, norm_layer=norm_layer,
                                act_func=act_func, conv_layer=t_conv,
                                n_groups=nred_channels, bias=bias)     #instead of max-pooling
                        )

            conv = [
                    conv_block(pre_channels + nred_channels, nred_channels,
                        norm_layer=norm_layer, act_func=act_func,
                        conv_layer=conv_layer, use_res=use_res, bias=bias,
                        **kwargs)
                ]
            conv.extend([
                    conv_block(nred_channels, nred_channels,
                        norm_layer=norm_layer, act_func=act_func,
                        conv_layer=conv_layer, use_res=use_res, bias=bias,
                        **kwargs
                    )
                    for _ in range(n_convs - 2)
                ])
            conv.append(conv_block(nred_channels, nout_channels,
                        norm_layer=norm_layer, act_func=act_func,
                        conv_layer=conv_layer, use_res=use_res, bias=bias,
                        **kwargs))
            convs.append(nn.Sequential(*conv))
            x_channels = pre_channels

        self.convs = nn.ModuleList(convs)
        self.ups = nn.ModuleList(ups)

    def forward(self, inputs : List[Tensor]) -> Tensor:
        assert self.n_inputs == len(inputs)

        X = inputs[0]
        for i, pre in enumerate(inputs[1:]):
            aux = self.ups[i](X)
            X = self.convs[i](cat([pre, aux], dim=1))

        return X
