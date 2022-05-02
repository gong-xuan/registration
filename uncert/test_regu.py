import random
from collections import defaultdict
import itertools
import os
import argparse
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
from dataset import *
from torch.optim.lr_scheduler import StepLR
from train import *
from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='./u-b1e1k/78.2.ckpt', help='pth name of pre-trained model')
    parser.add_argument('--dataset', type=str, default='CANDI', help='CANDI for now')
    parser.add_argument('--bsize', type=int, default=1, help='Batch size') 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--logfile', default='', type=str)
    # parser.add_argument('--tweight', type=str, default='./b1e1k/79.11.ckpt')
    # parser.add_argument('--uncert', type=int, default=0)
    parser.add_argument('--droprate', type=float, default=0)
    # parser.add_argument('--grad-weight', type=float, default=0)
    # parser.add_argument('--sgd', action='store_true')
    # parser.add_argument('--feat', action='store_true')
    return parser.parse_args()

from uncert import seg_uncert

def test_uncert(model, test_dataloader, n_class):
    var = []
    gt = []
    pred = []
    for _, (fixed, fixed_label, moving, moving_label) in enumerate(test_dataloader):
        fixed = torch.unsqueeze(fixed,1).float().cuda()
        moving = torch.unsqueeze(moving,1).float().cuda()
        fixed_label = fixed_label.float().cuda()
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().cuda()
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=n_class).float().permute(0,4,1,2,3)
        # dice = model.forward(fixed, moving,  fixed_label, moving_label, rtloss=False,eval=True)
        # dice=dice.mean()
        import ipdb; ipdb.set_trace()
        warp_label, flow, flow_var = model.forward(fixed, moving,  fixed_label, moving_label, rtloss=False)
        pred, gt, var = warp_label, fixed_label, flow_var
        AUSE, AURG = seg_uncert(pred, gt, var, intervals = 50,  metrics=['acc','MR1','FDR1','MR2','FDR2'], savedir='')
        import ipdb; ipdb.set_trace()
        # var.append(flow_var.detach())
        # gt.append(fixed_label.detach())
        # pred.append(warp_label.detach())#argmax->onehot
    # import ipdb; ipdb.set_trace()
    # var = torch.cat(var)
    # gt = torch.cat(gt)
    # pred = torch.cat(pred) 
    # 
    # if savedir and (not os.path.exists(savedir)):
    #     os.mkdir(savedir)

    import ipdb; ipdb.set_trace()
        


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    print(f'GPU: gpu')

    if args.dataset=='CANDI':
        pad_size=[160,160,128]
        window_r = 7
        NUM_CLASS = 29
    _, test_dataloader = CANDI_dataloader(args, size=pad_size)
    #
    # tmodel = models.RegNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    # tmodel.load_state_dict(torch.load(args.tweight))
    model = models.RegUncertNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS, grad_weight=0).cuda()
    model.eval()
    model.load_state_dict(torch.load(args.weight))

    test_uncert(model, test_dataloader, NUM_CLASS)