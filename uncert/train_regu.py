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
from train import *
from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--pretrain', type=str, default=None, help='pth name of pre-trained model')
    parser.add_argument('--dataset', type=str, default='CANDI', help='CANDI for now')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weightdecay', type=float, default=0, help='Weightdecay')
    parser.add_argument('--epoch', type=int, default = 1000, help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=1, help='Batch size') 
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--savefrequency', type=int, default=1, help='savefrequency')
    # parser.add_argument('--testfrequency', type=int, default=1, help='testfrequency')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--tweight', type=str, default='./b1e1k/79.11.ckpt')
    # parser.add_argument('--weight', type=str, default='1,0.01', help='LAMBDA, GAMMA')
    # parser.add_argument('--uncert', type=int, default=0)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--droprate', type=float, default=0)
    parser.add_argument('--grad-weight', type=float, default=0)
    parser.add_argument('--sgd', action='store_true')
    # parser.add_argument('--feat', action='store_true')
    return parser.parse_args()




if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # args.weight = [float(i) for i in args.weight.split(',')]
    
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
    # device = torch.device("cuda")
    # logging.info(f'Device: {device}')
    
    logging.info(f"DEVICE COUNT {torch.cuda.device_count()}")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')

    if args.dataset=='CANDI':
        pad_size=[160,160,128]
        window_r = 7
        NUM_CLASS = 29
    train_dataloader, test_dataloader = CANDI_dataloader(args, size=pad_size)
    #
    tmodel = models.RegNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS).cuda()
    tmodel.load_state_dict(torch.load(args.tweight))
    model = models.RegUncertNet(pad_size, winsize = window_r, dim = 3, n_class=NUM_CLASS, grad_weight=args.grad_weight).cuda()
    if len(gpu)>1:
        model = nn.DataParallel(model, device_ids=gpu)
    #model = nn.DataParallel(model, device_ids=gpu).to(device)
    # import ipdb; ipdb.set_trace()
    
    
    # import ipdb; ipdb.set_trace()
    if not args.debug:
        writer_comment = f'{args.logfile}'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)
    else:
        tb = None

    
    
    trainmodel = TrainUncertModel(tmodel, model, train_dataloader, test_dataloader, args, NUM_CLASS, tb=tb)
    
    if not args.snapshot:
        trainmodel.run()
    else:
        trainmodel.run_snapshot(cycles=20)
        
    if tb is not None:
        tb.close()

    
    
    