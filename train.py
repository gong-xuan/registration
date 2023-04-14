import torch
import logging
from utils import *
import os
from losses import *
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from uncert.snapshot import CyclicCosAnnealingLR
from models import RegNet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from collections import OrderedDict
from test import TestModel

class TrainModel():
    def __init__(self, model, train_dataloader, test_dataloader, args, n_class, tb=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tb = tb
        self.args = args
        self.n_class = n_class
        self.printfreq=50
        self.logfile = args.logfile
        self.cur_epoch = 0
        self.cur_idx = 0
        self.save_seg = args.save_seg
        #
        if args.logfile:
            savepath = os.path.join(args.logfile, 'ckpts')
        else:
            savepath = f'ckpts/vm'
        # if not os.path.isdir(savepath):
        #     os.makedirs(savepath)
        self.savepath = savepath
        #
    def trainIter(self, fix, moving, fixed_label, moving_label, fixed_nopad=None, seg_f=None): 
        if seg_f is not None:
            segimg_logfile = "seg_imgs"
            if not os.path.exists(os.path.join(self.logfile, segimg_logfile)):
                os.makedirs(os.path.join(self.logfile, segimg_logfile))
            seg_f = os.path.join(self.logfile, segimg_logfile, "e") + str(self.cur_epoch) + "idx" + str(self.cur_idx) + "_" + seg_f
        sim_loss, grad_loss, seg_loss, dice = self.model.forward(fix, moving,
            fixed_label, moving_label, fix_nopad=fixed_nopad, rtloss=True,eval=True, seg_fname=seg_f)
        
        sim_loss, grad_loss, seg_loss = sim_loss.mean(), grad_loss.mean(), seg_loss.mean() 
        loss = float(self.args.weight[0])*sim_loss + float(self.args.weight[1])*grad_loss + float(self.args.weight[2])*seg_loss

        dice = dice.mean() # Mahesh : Why do we dice.mean again ?
        if self.global_idx%self.printfreq ==0:
            logging.info(f'simloss={sim_loss}, gradloss={grad_loss}, segloss={seg_loss}, loss={loss}, dice={(dice.item()*100):2f}')
        if self.tb is not None and dist.get_rank() == 0:
            self.tb.add_scalar("train/loss", loss.item(), self.global_idx)
            self.tb.add_scalar("train/grad_loss", grad_loss.item(), self.global_idx)
            self.tb.add_scalar("train/sim_loss", sim_loss.item(), self.global_idx)
            self.tb.add_scalar("train/seg_loss", seg_loss.item(), self.global_idx)
            self.tb.add_scalar("train/Dice", dice.item(), self.global_idx)
        return loss,dice

    def data_extract(self, samples, device=torch.device(type='cuda', index=0)):
        seg_fname = samples[-1][0]
        samples = samples[:-1]
        if len(samples)==4:
            fixed, fixed_label, moving, moving_label = samples
            fixed_nopad = None
        elif len(samples)==5:
            fixed, fixed_label, fixed_nopad, moving, moving_label = samples  
        elif len(samples) == 6:
            fixed, fixed_label, moving, moving_label, _, _ = samples    
            fixed_nopad = None      
        #
        fixed = torch.unsqueeze(fixed,1).float().to(device)
        moving = torch.unsqueeze(moving,1).float().to(device)
        fixed_label = fixed_label.float().to(device)
        # import ipdb; ipdb.set_trace()
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().to(device)
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        if fixed_nopad is not None:
            fixed_nopad = fixed_nopad.to(device)[:, None]
        return fixed, fixed_label, moving, moving_label, fixed_nopad, seg_fname

    def test(self, epoch):
        tst_dice = AverageMeter()
        self.model.eval()
        idx = 0
        for _, samples in enumerate(self.test_dataloader):
            if self.model.__class__.__module__ == 'torch.nn.parallel.distributed':
                fixed, fixed_label, moving, moving_label, fixed_nopad, seg_fname = self.data_extract(samples, device=self.model.device)
            else:
                fixed, fixed_label, moving, moving_label, fixed_nopad, seg_fname = self.data_extract(samples)
            if not self.save_seg: # Remove this if you want to get the segmentation images on the test volumes
                seg_fname = None
            with torch.no_grad():
                dice = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True, seg_fname=seg_fname)
                # dice2 = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True)
            # logging.info(f'itration={idx}/{len(self.test_dataloader)}')
            # logging.info(f'test dice={(dice.item()*100):2f}')
            idx+=1
            dice = dice.mean()
            tst_dice.update(dice.item())
        #epoch
        if self.tb is not None and dist.get_rank() == 0:
            self.tb.add_scalar("test/dice", tst_dice.avg, epoch)
        return tst_dice.avg
    
    def train_epoch(self, optimizer, scheduler, epoch):
        epoch_train_dice = AverageMeter()
        self.model.train()
        idx = 0
        for n_iter, samples in enumerate(self.train_dataloader):
            fixed, fixed_label, moving, moving_label, fixed_nopad, seg_fname = self.data_extract(samples, self.model.device)
            if not self.save_seg:
                seg_fname = None
            self.global_idx += 1
            self.cur_idx = idx
            logging.info(f'iteration={idx}/{len(self.train_dataloader)}')
            idx+=1
            loss, trdice = self.trainIter(fixed, moving, fixed_label, moving_label, fixed_nopad=fixed_nopad, seg_f=seg_fname)
            optimizer.zero_grad()
            loss.backward()
            # import ipdb; ipdb.set_trace()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                    
            epoch_train_dice.update(trdice.item())       
        if self.tb is not None and dist.get_rank() == 0:
            self.tb.add_scalar("train/dice", epoch_train_dice.avg, epoch)
        
        #validate per epoch
        # import ipdb; ipdb.set_trace()
        
        if self.test_dataloader is not None and dist.get_rank() == 0:
            dice = self.test(epoch)
            # crtest_dice = -1
            if dice>self.bestdice:
                self.bestdice = dice
                savename = os.path.join(self.savepath, f'{epoch}_{(dice*100):2f}.pth')
                torch.save(self.model.state_dict(), savename)
                # test = TestModel(savename, [320, 320, 176], 5, self.test_dataloader, 2, True)
                # crtest_dice = test.test()
                # if crtest_dice != (dice):
                #     print("Discrepency!!")
            logging.info(f'Epoch:{epoch}...TestDice:{(dice*100):2f}, Best{(self.bestdice*100):2f}')

    def run(self):#device=torch.device("cuda:0")
        if self.args.sgd:
            optimizer = SGD(self.model.parameters(), lr = self.args.lr)
        else:
            optimizer = Adam(self.model.parameters(),lr = self.args.lr)

        # scheduler = StepLR(opt, step_size=5, gamma=0.1)
        scheduler = None

        self.global_idx = 0
        self.bestdice=0
        for epoch in range(self.args.epoch):
            self.train_epoch(optimizer, scheduler, epoch)
            self.cur_epoch = epoch
    
    def run_snapshot(self, cycles=20, ):
        epochs = self.args.epoch
        epochs_per_cycle = epochs// cycles
        global_epoch=0
        self.global_idx = 0
        self.bestdice=0

        for n_cycle in range(cycles):
            if self.args.sgd:
                optimizer = SGD(self.model.parameters(), lr = 0.001)#0.01 lossNan
            else:
                optimizer = Adam(self.model.parameters(), lr = self.args.lr)#1e-4
            
            scheduler = CyclicCosAnnealingLR(optimizer, epochs_per_cycle)
            for epoch in range(epochs_per_cycle):
                global_epoch+=1
                self.train_epoch(optimizer, scheduler, global_epoch)

            savename = os.path.join(self.savepath, f'cycle{n_cycle}_{(self.bestdice*100):2f}.ckpt')
            torch.save(self.model.state_dict(), savename)

"""
class TrainUncertModel(TrainModel):
    def __init__(self, tmodel, model, train_dataloader, test_dataloader, args, n_class, tb=None):
        self.tmodel = tmodel
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tb = tb
        self.args = args
        self.n_class = n_class
        self.printfreq=50
        #
        if args.logfile:
            savepath = args.logfile
        else:
            savepath = './uncert'
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.savepath = savepath

    def trainIter(self, fix, moving, fixed_label, moving_label, ):
        gtflow = self.tmodel.forward(fix, moving,  fixed_label, moving_label, rtloss=False,eval=False).detach()
        loss, dice = self.model.forward(fix, moving,  fixed_label, moving_label, gtflow=gtflow, rtloss=True,eval=True)
        
        if self.global_idx%self.printfreq ==0:
            logging.info(f'loss={loss.item()}, dice={dice.item()}')
        if self.tb is not None:
            self.tb.add_scalar("train/loss", loss.item(), self.global_idx)
        return loss,dice
"""