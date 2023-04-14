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
from dataset import candi, msd, brats, chaos, learn2reg
from torch.optim.lr_scheduler import StepLR
from train import TrainModel
from models import RegNet
# from raft import RaftRegNet
# from msraft import MsRaftRegNet
# from tsraft import TsRaftRegNet
import math
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing, logging
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Pool


CANDI_PATH = '/a2il/data/mbhosale//CANDI_split'
MSD_PATH = '/a2il/data/mbhosale/MSD/'
# MSD_PATH = '/a2il/data/mbhosale/MSD_resampled/' # for clon and spleen datasets use resampled path
# BraTS_PATH = '/data_local/xuangong/data/BraTS'
CHAOS_PATH = r'/a2il/data/mbhosale/CHAOS_preprocessed/'
L2R_DATAPATH = r'/a2il/data/mbhosale/learn2reg/AbdomenMRCT/'

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
    parser.add_argument('--port', default='12345', type=str, help='Port for vommunivating multi-gpu proccessing')
    parser.add_argument('--logfile', default='./logs/', type=str)
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
    parser.add_argument('--mod', type=str, default='t1') #modality for BraTS: t1, t2, t1ce, flair
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--tr_modality', type=str, default='T1DUAL', help='train modality')
    parser.add_argument('--tst_modality', type=str, default='T1DUAL', help='test modality')
    parser.add_argument('--tr_phase', type=str, default='InPhase', help='train phase')
    parser.add_argument('--tst_phase', type=str, default='InPhase', help='test phase')
    parser.add_argument('--save_config', type=bool, default=False)
    parser.add_argument('--use_config', type=bool, default=True)
    parser.add_argument('--save_seg', type=bool, default=False)
    return parser.parse_args()

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_logger(args, window_r):
    if args.debug:
        logfile = os.path.join(args.logfile, "logs_window_{}.txt".format(window_r))
    else:
        logfile = os.path.join(args.logfile, f'{datetime.now().strftime("%m%d%H%M")}.txt')
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(
        logfile, mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    # this bit will make sure you won't have 
    # duplicated messages in the output
    # if not len(logger.handlers): 
    #     logger.addHandler(handler)
    return logger

def run_parallel(rank, pad_size, window_r, NUM_CLASS, train_dataloader, test_dataloader, args, world_size):
    import time
    start = time.process_time()
    # your code here    
    logger = create_logger(args, window_r)
    globals()['logger'] = logger
    logger.info('Starting pooling')
    p = Pool()
    logging.info(f"Running basic DDP example on rank {rank}.")
    logging.info(args)

    logging.info(f"DEVICE COUNT {torch.cuda.device_count()}")
    logging.info(f'GPU: {args.gpu}')
    # print(f"Rank {rank} Time taken 1:" + str(time.process_time() - start))
    
    # start = time.process_time()
    setup(rank, world_size, args.port)
    # print(f"Rank {rank} Time taken 2:" + str(time.process_time() - start))    

    writer_logdir = os.path.join(args.logfile, 'tb')#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
    tb = SummaryWriter(log_dir= writer_logdir)

    ##BUILD MODEL##
    # start = time.process_time()

    model = RegNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS, feat=args.feat).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device = rank)
    train = TrainModel(ddp_model, train_dataloader, test_dataloader, args, NUM_CLASS, tb=tb)
    # print(f"Rank {rank} Time taken 3:" + str(time.process_time() - start))
    train.run()
    if tb is not None:
        tb.close()
    cleanup()
    
def correct_padsize(pad_size, downsample_rate):
    if pad_size[-1]%downsample_rate != 0:               
            orig_size = pad_size
            c_dim = orig_size[-1]
            pad_size[-1] += abs(c_dim - (math.ceil(c_dim/downsample_rate)*downsample_rate))
    return pad_size
    
if __name__ == "__main__":
    args = get_args()
    torch.multiprocessing.set_sharing_strategy('file_system') # handle large number of files. Increase the number of open files.
    torch.cuda.empty_cache()
    downsample_rate = 16
    args.use_config = False
    args.save_config = False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.weight = [float(i) for i in args.weight.split(',')]
    
    if args.logfile:
        args.logfile = os.path.join(args.logfile, args.dataset, datetime.now().strftime("%m_%d_%y_%H_%M"))
        if not os.path.isdir(args.logfile):
            os.makedirs(args.logfile)
    if args.logfile:
        savepath = os.path.join(args.logfile, 'ckpts')
    else:
        savepath = f'ckpts/vm'
    if not os.path.exists(savepath):
            os.makedirs(savepath)
    #load model
    #device = torch.device(0)
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    world_size = len(gpu)
    if args.dataset=='CANDI':
        pad_size=[160,160,128]
        window_r = 7
        NUM_CLASS = 29
        train_dataloader, test_dataloader = candi.CANDI_dataloader(args, datapath=CANDI_PATH, size=pad_size)
    elif args.dataset in ['prostate', 'hippocampus', 'liver', 'heart', 'spleen', 'colon']:
        if args.dataset=='prostate':
            NUM_CLASS = 3
            window_r = 9
            pad_size = [240, 240, 96] 
        elif args.dataset == 'hippocampus': 
            NUM_CLASS = 3
            window_r = 5
            pad_size = [48, 64, 48]
        elif args.dataset == 'liver':
            NUM_CLASS = 3
            window_r = 9
            pad_size = [256, 256, 128]
        elif args.dataset == 'heart':
            NUM_CLASS = 2
            window_r = 5
            pad_size = [320, 320, 176]
        elif args.dataset == 'spleen':
            NUM_CLASS = 2
            window_r = 5
            pad_size = [256, 256, 64]
        elif args.dataset == 'colon':
            NUM_CLASS = 2
            window_r = 5
            pad_size = [256, 256, 64]
        pad_size = correct_padsize(pad_size=pad_size, downsample_rate=downsample_rate)
        train_dataloader, test_dataloader, _= msd.MSD_dataloader(args.dataset, args.bsize, pad_size, args.num_workers, datapath=MSD_PATH, tr_percent=1, testseg=1, testreg=0, save_config=args.save_config, use_config=args.use_config)
    elif args.dataset=='brats': #not proper for registration?
        pad_size = [240, 240, 160]
        NUM_CLASS = 4 #including background 
        train_dataloader, test_dataloader = brats.BraTS_dataloader(args, root_path=BraTS_PATH, size = pad_size)
        # pad_size = train_dataloader.dataset.size        
        window_r = 7
    elif args.dataset == 'chaos':
        # # Mahesh : The x-y size of the images is caculated based on maximum z-y size of the dicom images.
        # pad_size = [400, 400, 50]
        # tr_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR/"
        # tst_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR/"
        # train_dataloader, test_dataloader = chaos.Chaos_dataloader(root_path=CHAOS_PATH,  tr_path=tr_path, tst_path=tst_path, 
        #                                                  bsize=1, tr_modality='T1DUAL', tr_phase='InPhase', tst_modality='T1DUAL', 
        #                                                  tst_phase='InPhase', size=[400, 400, 50], data_split=False, n_fix=1, tr_num_samples=0, 
        #                                                  tst_num_samples=10) 
        # window_r = 11
        # # Mahesh : Should the mumber of classes be one more than total number of classes? As required for some of the loss functions etc. >> No Need, 
        # # we are not using any other loss function such as cross entropy loss which takes in number of classes as an arguemnt.
        # NUM_CLASS = 5
        
        pad_size = [256, 256, 64] # for T1DUAL [256, 256, 50] # TODO Why is it 320 for T2spir ? 
        tr_path = CHAOS_PATH + r"/CHAOS_Train_Sets/Train_Sets/MR/"
        tst_path = CHAOS_PATH + r"/CHAOS_Val_sets/Val_sets/MR/" # we are choosing train dataset as test because we dont have ground truth in the test
        # TODO But we can change the test modality, but for now we have kept it same 
        train_dataloader, test_dataloader = chaos.Chaos_dataloader(root_path=CHAOS_PATH,  tr_path=tr_path, tst_path=tst_path, 
                                                         bsize=1, tr_modality=args.tr_modality, tr_phase=args.tr_phase, tst_modality=args.tst_modality, 
                                                         tst_phase=args.tst_phase, size=pad_size, data_split=False, n_fix=1)
        if pad_size[-1]%downsample_rate != 0:               
            orig_size = pad_size
            c_dim = orig_size[-1]
            pad_size[-1] += abs(c_dim - (math.ceil(c_dim/downsample_rate)*downsample_rate))
        window_r = 5
        NUM_CLASS = 5
    elif args.dataset == "learn2reg":
        pad_size = [192, 160, 192]
        train_dataloader, test_dataloader = learn2reg.l2r_dataloader(datapath=L2R_DATAPATH, 
                                                                     size=pad_size, mod="MR", bsize=1, num_workers=1)
        window_r = 9
        NUM_CLASS = 5
        
    ##BUILD MODEL##
    # if args.tsraft:
    #     down_flatten = 32//(2**args.down)
    #     model = TsRaftRegNet(size=pad_size, corr_radius=args.corr, iters=args.iters, corr_levels=1,  downsample=args.down,
    #         down_flatten=down_flatten, upconv=args.upconv, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    # elif args.msraft:
    #     model = MsRaftRegNet(size=pad_size, corr_radius=args.corr, winsize = window_r, ks_loss=args.ksloss, mode=args.msraft, n_class=NUM_CLASS).cuda()
    # elif args.raft:
    #     model = RaftRegNet(size=pad_size, corr_radius=args.corr, iters=1, downsample=3, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    # else:
    mp.spawn(run_parallel,
             args=(pad_size, window_r, NUM_CLASS, train_dataloader, test_dataloader, args, world_size),
             nprocs=world_size,
             join=True)

    
