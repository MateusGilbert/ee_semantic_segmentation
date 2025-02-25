#! /usr/bin/python3

from conv_blocks import *
from build_unet import *
from from_unet import *
#from from_unet_2 import *
#from build_unet_2 import *
from build_ae import *
#from seg_losses import *
from seg_funcs_tch import *
from new_seg_funcs import *
from new_seg_losses import *
from seg_metrics import *
from torch import nn
from torch.nn import init
#from sklearn.metrics import mean_squared_error as MSE,\
# top_k_accuracy_score, precision_score as precision,\
# recall_score as recall, f1_score, accuracy_score as accuracy
#from skimage.metrics import structural_similarity as SSIM,\
# normalized_root_mse as NRMSE, peak_signal_noise_ratio as PSNR,\
# mean_squared_error as Img_MSE, hausdorff_distance as hausdorff
from torch import round as tch_round
import branchy_seg_losses as BSL
from eval_mIoU import mIoU_evaluator

conv_blks = {
    'InvertedResidual': InvertedResidual,
    'InceptionBlk': InceptionBlk,
    'CIncepBlk': CIncepBlk,
    'HDConvBlk': HDConvBlk,
    'WaveBlk': WaveBlk,
}

get_mod = {
    'DenseLayer': DenseLayer,
    'ConvLayer': ConvLayer,
} | conv_blks

net_consts = {
    'UNet': UNet,
#    'UNet_2': UNet_2, #gambiarra
    'EUNet': EUNet,
    'AE': AE,
    'branchy_unet': branchy_unet,
#    'branchy_unet_2': branchy_unet_2,
}

#default initialization
seg_losses = {
    'FocalLoss': FocalLoss(),
    'FocalLoss_sum': FocalLoss(reduction='sum'),
    'JaccardLoss': JaccardLoss(),
    'JaccardLoss_sum': JaccardLoss(reduction='sum'),
    'JaccardLoss_sum_dg': JaccardLoss(reduction='sum', downgrad_bg=.05),
    'TverskyLoss': TverskyLoss(alpha=.7, beta=.3),
    'TverskyLoss_sum': TverskyLoss(alpha=.7, beta=.3, reduction='sum'),
    'FocalTverskyLoss': FocalTverskyLoss(alpha=.7, beta=.3, gamma=4/3),
    'FocalTverskyLoss_sum': FocalTverskyLoss(alpha=.7, beta=.3, gamma=4/3, reduction='sum'),
    'HybridFocal': HybridFocalLoss(),
    'LovaszSoftmax': LovaszSoftmax(ignore=21),
    'LovaszSoftmax_ignore': LovaszSoftmax(ignore=0),        #ignore background
    #definir n_branches
    'BSL_Focal': BSL.FocalLoss(n_branches=4),
    'BSL_Jaccard': BSL.JaccardLoss(n_branches=4),
    'BSL_Tversky': BSL.TverskyLoss(alpha=.7, beta=.3, n_branches=4),#, weights=list(map(lambda x: x/5, range(1,6)))),
    'BSL_FocalTversky': BSL.FocalTverskyLoss(alpha=.7, beta=.3, gamma=4/3, n_branches=4),#, weights=list(map(lambda x: x/5, range(1,6)))),
}

get_loss = {
    'MSE': nn.MSELoss(),
    'MSE_sum': nn.MSELoss(reduction='sum'),
    'L1': nn.L1Loss(),
    'S_L1': nn.SmoothL1Loss(),  #see https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    'x_entropy': nn.CrossEntropyLoss(),
    'x_entropy_sum': nn.CrossEntropyLoss(reduction='sum'),
    'nll': nn.NLLLoss(),
    'ml_softmargin': nn.MultiLabelSoftMarginLoss(),
} | seg_losses

#Defining top K before
#class TopKAcc:
#    def __init__(self,k=2):
#        self.k = 2
#    def __call__(self,y,y_pred):
#        return top_k_accuracy_score(y,y_pred,k=self.k)

#class pixel_acc:
#    def __init__(self, round=True):
#        self.round = round
#
#    def __call__(self, y, y_pred):
#        return accuracy(y.flatten(), tch_round(y_pred).flatten() if self.round else y_pred.flatten())


get_metric = {
#    'MSE': MSE,
#    'Img_MSE': Img_MSE,
#    'SSIM': SSIM,
#    'PSNR': PSNR,
#    'NRMSE': NRMSE,
#    'accuracy': Accuracy(),
#    'precision': Precision(),
#    'recall': Recall(),
#    'Top3acc': TopKAcc(3),
#    'Top5acc': TopKAcc(5),
#    'Top7acc': TopKAcc(7),
    'F1': F_beta(),
    'F2': F_beta(beta=2),
    'F.5': F_beta(beta=.5),
    'Dice': DiceLoss(index=True),
    'Jaccard': JaccardLoss(index=True),
    'mIoU': mIoU_evaluator,
#    'img_acc': accuracy_images,
#    'img_rec': recall_images,
#    'img_prc': precision_images,
#    'img_f1': f1_score_images,
#    'jacc_idx': jaccard_index,
#    'hausdorff': hausdorff,
#    'seg_score': seg_score,
#    'dice_idx': dice_index,
} | seg_losses

get_initializer = {
    'xavier_u': init.xavier_uniform_,
    'xavier_n': init.xavier_normal_,
    'dirac': init.dirac_,    #for {3,4,5}-dimensional tensor
    'normal': init.normal_,
    'uniform': init.uniform_,
    'ones': init.ones_,
    'orthogonal': init.orthogonal_,
    'kaiming_uniform': init.kaiming_uniform_,
    'kaiming_normal': init.kaiming_normal_,
}

get_conv = {
    '1d': nn.Conv1d,
    '2d': nn.Conv2d,
    '3d': nn.Conv3d
}

#auxiliary variables
act_funcs = (
        nn.ReLU,
        nn.SELU,
        nn.LeakyReLU,
        nn.ELU,
        nn.CELU,
        nn.Mish,
        nn.ReLU6,
)
