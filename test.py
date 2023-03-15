import torch
import logging
from utils import *
import os
from losses import *
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from uncert.snapshot import CyclicCosAnnealingLR
from datetime import datetime
import dataset.msd as msd
import dataset.candi as candi
import argparse
from models import RegNet
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from utils import *

# Test datapaths
CANDI_PATH = r'/data_local/mbhosale/CANDI_split'

# colon, spleen
# MSD_PATH = r'/a2il/data/mbhosale/MSD_resampled/'

# liver, hippocampus, prostate. heart
MSD_PATH = r'/a2il/data/mbhosale/MSD/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='prostate', help='CANDI, prostate, hippocampus, liver, colon, liver, spleen, learn2reg, chaos')
    parser.add_argument('--test_dataset', type=str, default='CANDI', help='CANDI, prostate, hippocampus, liver, colon, liver,  learn2reg, chaos')
    parser.add_argument('--log', default='./logs/cross_domain_tests/', type=str)
    parser.add_argument('--bsize', type=int)
    parser.add_argument('--gpu', default='0', type=str, help='GPU to use for inference')
    parser.add_argument('--checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--feat', action='store_true')
    return parser.parse_args()

class TestModel():
    def __init__(self, ckpt_file, padsize, win_size, test_dataloader, n_class, feat):
        self.test_dataloader = test_dataloader
        self.n_class = n_class
        model = RegNet(padsize, winsize=win_size, dim=3, n_class=n_class, feat=feat).cuda()
        # d = torch.load(ckpt_file)
        # new_state_dict = OrderedDict()
        # for k, v in d.items():
        #     name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        #     new_state_dict[name] = v
        # d = new_state_dict
        # d = {k: v for k, v in d.items() if k !='spatial_transformer_network.meshgrid'}
        # model.load_state_dict(d, strict=False)
        model = load_dict(ckpt_file, model)
        self.model = model

    def data_extract(self, samples):
        # doesnt support multi GPU tests
        if len(samples)==4:
            fixed, fixed_label, moving, moving_label = samples
            fixed_nopad = None
        elif len(samples)==5:
            fixed, fixed_label, fixed_nopad, moving, moving_label = samples  
        elif len(samples) == 6:
            fixed, fixed_label, moving, moving_label, _, _ = samples    
            fixed_nopad = None  
            
        moving = torch.unsqueeze(moving, 1).float().cuda()
        fixed = torch.unsqueeze(fixed, 1).float().cuda()
        fixed_label = fixed_label.float().cuda()
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().cuda()
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)

        if fixed_nopad is not None:
            fixed_nopad = fixed_nopad.float().cuda()[:, None]
            fixed_label = fixed_nopad * fixed_label
        return fixed, fixed_label, moving, moving_label, fixed_nopad
    
    # def test(self):
    #     tst_dice = AverageMeter()
    #     self.model.eval()
    #     logging.info(" started")
    #     idx = 0
    #     for _, samples in enumerate(self.test_dataloader):
    #         fixed, fixed_label, moving, moving_label, fixed_nopad = self.data_extract(samples)
    #         with torch.no_grad():
    #             dice = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True)
    #         dice = dice.mean()
    #         tst_dice.update(dice.item())
    #         logging.info(f'iteration={idx}/{len(self.test_dataloader)}')
    #         logging.info(f'test dice={dice.item()}')
    #         idx+=1
    #     logging.info(f'Average test dice= {tst_dice.avg}')
    
    def test(self):
        tst_dice = AverageMeter()
        self.model.eval()
        idx = 0
        torch.save(self.model.state_dict(), 'test_checkpoint1.pth')
        for _, samples in enumerate(self.test_dataloader):
            fixed, fixed_label, moving, moving_label, fixed_nopad = self.data_extract(samples)
            with torch.no_grad():
                dice = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True)
                # dice2 = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True)
            dice = dice.mean()
            logging.info(f'iteration={idx}/{len(self.test_dataloader)}')
            logging.info(f'test dice={(dice.item()*100):2f}')
            idx+=1
            tst_dice.update(dice.item())
        torch.save(self.model.state_dict(), 'test_checkpoint2.pth')
        logging.info(f'Average test dice= {(tst_dice.avg*100):2f}')
        return tst_dice.avg

if __name__ == "__main__":
    args = get_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.isdir(args.log):
            os.makedirs(args.log)
    logfile = os.path.join(args.log, "train_"+ args.train_dataset + "_test_" + args.test_dataset + "_" + datetime.now().strftime("%m_%d_%y_%H_%M")+".txt")
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(
        logfile, mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    
    # assert(args.train_dataset != args.test_dataset)
    # Set things required for the testing
    if args.test_dataset in ['prostate', 'hippocampus', 'liver', 'heart', 'colon', 'spleen']:
        test_datapath = MSD_PATH
        if args.test_dataset == "hippocampus":
            tst_pad_size = [48, 64, 48]
            window_r = 5
        elif args.test_dataset == "liver":
            tst_pad_size = [256, 256, 128]
            window_r = 9
        elif args.test_dataset == "prostate":
            tst_pad_size = [240, 240, 96]
            window_r = 9
        elif args.test_dataset == "heart":
            tst_pad_size = [320, 320, 176]
            window_r = 5
        elif args.test_dataset == 'colon':
            tst_pad_size = [256, 256, 64]
            window_r = 5
        else:
            tst_pad_size = [256, 256, 64]
            window_r = 5
        _, testseg_dataloader, _ = msd.MSD_dataloader(args.test_dataset, args.bsize, tst_pad_size, args.num_workers, datapath=test_datapath, tr_percent=1, testseg=1, testreg=0)
    elif args.test_dataset == "CANDI":
        test_datapath = CANDI_PATH
        tst_pad_size=[160, 160, 128]
        _, testseg_dataloader = candi.CANDI_dataloader(args, size=tst_pad_size, datapath=test_datapath)
        
    # Set things required for loading the checkpoint of the trained dataset.
    if args.train_dataset=='CANDI':
        NUM_CLASS = 29
        tr_pad_size = [160,160,128]
    elif args.train_dataset in ['prostate', 'hippocampus', 'liver', 'heart', 'colon', 'spleen']:
        if args.train_dataset == 'prostate':
            NUM_CLASS = 3
            tr_pad_size = [240, 240, 96] 
        elif args.train_dataset == 'hippocampus':
            NUM_CLASS = 3
            tr_pad_size = [48, 64, 48]
        elif args.train_dataset == 'liver':
            NUM_CLASS = 3
            tr_pad_sze = [256, 256, 128]
        elif args.train_dataset == 'heart':
            NUM_CLASS = 2
            tr_pad_size = [320, 320, 176]
        elif args.train_dataset == 'colon':
            NUM_CLASS = 2
            tr_pad_size = [256, 256, 64]
        else:
            NUM_CLASS = 2
            tr_pad_size = [256, 256, 64]
    test = TestModel(args.checkpoint, tst_pad_size, window_r, testseg_dataloader, NUM_CLASS, args.feat)
    
    # # Save and test the loaded the model. #TODO just for debug, remove it later.
    # savename = 'checkpoint1.ckpt'
    # torch.save(test.model.state_dict(), savename)
    
    # model1 = RegNet([240, 240, 96] , winsize=test.model.winsize, dim=3, n_class=test.model.n_class).to(0)
    # d = torch.load(savename)
    # # Skip loading the spatial transformer, as the train and test datasets are different.
    # d = {k: v for k, v in d.items() if k !='spatial_transformer_network.meshgrid'}
    # model1.load_state_dict(d, strict=False)
    
    test.test()
    
    # Save and test the loaded the model. #TODO just for debug, remove it later.
    # savename = 'checkpoint2.ckpt'
    # torch.save(test.model.state_dict(), savename)
    
    # model2 = RegNet([240, 240, 96] , winsize=test.model.winsize, dim=3, n_class=test.model.n_class).to(0)
    # d = torch.load(savename)
    # # Skip loading the spatial transformer, as the train and test datasets are different.
    # d = {k: v for k, v in d.items() if k !='spatial_transformer_network.meshgrid'}
    # model2.load_state_dict(d, strict=False)
    # print("hi")