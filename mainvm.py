import random
from collections import defaultdict
import itertools
import os
import argparse
import logging
from datetime import datetime
import torch
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
# import nibabel as nib
from dataset import candi, msd, brats, chaos
from torch.optim.lr_scheduler import StepLR
from train import TrainModel
from models import RegNet
from raft import RaftRegNet
from msraft import MsRaftRegNet
from tsraft import TsRaftRegNet


CANDI_PATH = '/data_local/xuangong/data/CANDI_split'
MSD_PATH = '/a2il/data/MedReg/MSD'
BraTS_PATH = '/data_local/xuangong/data/BraTS'
CHAOS_PATH = r'/data_local/mbhosale/CHAOS/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--pretrain', type=str, default=None, help='pth name of pre-trained model')
    parser.add_argument('--dataset', type=str, default='CANDI', help='CANDI for now')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weightdecay', type=float, default=0, help='Weightdecay')
    parser.add_argument('--epoch', type=int, default = 500, help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=1, help='Batch size') 
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--savefrequency', type=int, default=1, help='savefrequency')
    # parser.add_argument('--testfrequency', type=int, default=1, help='testfrequency')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--weight', type=str, default='1,0.01,1', help='weight for imgsim, grad, segsim')
    # parser.add_argument('--uncert', type=int, default=0)
    # parser.add_argument('--dual', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--feat', action='store_true')
    parser.add_argument('--droprate', type=float, default=0)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--raft', action='store_true')
    parser.add_argument('--msraft', type=str, default='')
    parser.add_argument('--tsraft', action='store_true')
    parser.add_argument('--ksloss', type=float, default=0)
    parser.add_argument('--corr', type=int, default=3)
    parser.add_argument('--down', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--upconv', action='store_true')
    # parser.add_argument('--feat', action='store_true')
    parser.add_argument('--mod', type=str, default='t1') #modality for BraTS: t1, t2, t1ce, flair
    parser.add_argument('--eval', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.weight = [float(i) for i in args.weight.split(',')]
    
    handlers = [logging.StreamHandler()]
    if args.debug:
        logfile = f'debug'
    else:
        logfile = f'{args.logfile}-{datetime.now().strftime("%m%d%H%M")}'
    handlers.append(logging.FileHandler(
        f'./logs/{logfile}.txt', mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    #load model
    #device = torch.device(0)
    device = torch.device("cuda")
    logging.info(f'Device: {device}')
    
    logging.info(f"DEVICE COUNT {torch.cuda.device_count()}")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')

    if args.dataset=='CANDI':
        pad_size=[160,160,128]
        window_r = 7
        NUM_CLASS = 29
        train_dataloader, test_dataloader = candi.CANDI_dataloader(args, datapath=CANDI_PATH, size=pad_size)
    elif args.dataset in ['prostate', 'hippocampus', 'liver']:
        NUM_CLASS = 3
        if args.dataset=='prostate':
            window_r = 9
            pad_size = [240,240,96] 
        elif args.dataset == 'hippocampus': 
            window_r = 5
            pad_size = [48,64,48]
        elif args.dataset == 'liver':
            window_r = 15
            pad_size = [256, 256, 128]
        train_dataloader, test_dataloader, _= msd.MSD_dataloader(args.dataset, args.bsize, pad_size, args.num_workers, datapath=MSD_PATH, tr_percent=0.1, testseg=0, testreg=0)
        
    elif args.dataset=='brats': #not proper for registration?
        pad_size = [240, 240, 160]
        NUM_CLASS = 4 #including background 
        train_dataloader, test_dataloader = brats.BraTS_dataloader(args, root_path=BraTS_PATH, size = pad_size)
        # pad_size = train_dataloader.dataset.size        
        window_r = 7
    elif args.dataset == 'chaos':
        # Mahesh : The x-y size of the images is caculated based on maximum z-y size of the dicom images.
        pad_size = [400, 400, 50]
        tr_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR/"
        tst_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR/"
        train_dataloader, test_dataloader = chaos.Chaos_dataloader(root_path=CHAOS_PATH,  tr_path=tr_path, tst_path=tst_path, 
                                                         bsize=1, tr_modality='T1DUAL', tr_phase='InPhase', tst_modality='T1DUAL', 
                                                         tst_phase='InPhase', size=[400, 400, 50], data_split=False, n_fix=1, tr_num_samples=0, 
                                                         tst_num_samples=10) 
        window_r = 11
        # Mahesh : Should the mumber of classes be one more than total number of classes? As required for some of the loss functions etc. >> No Need, 
        # we are not using any other loss function such as cross entropy loss which takes in number of classes as an arguemnt.
        NUM_CLASS = 5

    ##BUILD MODEL##
    if args.tsraft:
        down_flatten = 32//(2**args.down)
        model = TsRaftRegNet(size=pad_size, corr_radius=args.corr, iters=args.iters, corr_levels=1,  downsample=args.down,
            down_flatten=down_flatten, upconv=args.upconv, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    elif args.msraft:
        model = MsRaftRegNet(size=pad_size, corr_radius=args.corr, winsize = window_r, ks_loss=args.ksloss, mode=args.msraft, n_class=NUM_CLASS).cuda()
    elif args.raft:
        model = RaftRegNet(size=pad_size, corr_radius=args.corr, iters=1, downsample=3, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    else:
        model = RegNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS, feat=args.feat).cuda()
    if len(gpu)>1: #only valid on deepbull/deepbull2
        model = torch.nn.DataParallel(model, device_ids=gpu)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpu)
        #model = nn.DataParallel(model, device_ids=gpu).to(device)
        # import ipdb; ipdb.set_trace()
    
    # if args.sgd:
    #     opt = SGD(model.parameters(), lr = args.lr)
    # else:
    #     opt = Adam(model.parameters(),lr = args.lr)

    # # scheduler = StepLR(opt, step_size=5, gamma=0.1)
    # scheduler = None
    # import ipdb; ipdb.set_trace()
    
    if not args.debug:
        writer_comment = f'{args.logfile}'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)
    else:
        tb = None

    train = TrainModel(model, train_dataloader, test_dataloader, args, NUM_CLASS, tb=tb)
    train.run()

    if tb is not None:
        tb.close()

    
