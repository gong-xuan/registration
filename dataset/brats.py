import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel
from dataset.msd import pad
import logging

class BraTSDataset(Dataset):
    def __init__(self, datapath, mod, size):
        """
        mode: t1, t2, t1ce, flair
        """
        self.datapath = datapath
        self.size = size
        self.mod = mod
        self.label = [1,2,4] 

        #split fix and others
        datapath = os.path.expanduser(datapath)
        sublist = sorted(os.listdir(datapath))
        fixsub = sublist[0]
        self.fiximg, self.fix_nopad = self.preprocess_img(f'{datapath}/{fixsub}/{fixsub[4:]}_{mod}.nii.gz')
        self.fixseg = self.preprocess_seg(f'{datapath}/{fixsub}/{fixsub[4:]}_seg.nii.gz')
        
        self.movingsub = sublist[1:]
        

    def __len__(self):
        return len(self.movingsub)

    def preprocess_img(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        #normalize
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        #std_arr = np.sqrt(np.abs(x-mean)/x.size)
        maxp = mean + 6*std
        minp = mean - 6*std
        y = np.clip(data, minp, maxp)
        #import ipdb; ipdb.set_trace()
        #linear transform to [0,1]
        z = (y-y.min())/y.max()
        z, nopad = pad(z, self.size)
        return z, nopad

    def preprocess_seg(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        #filter label
        seg = np.zeros_like(data)
        for n,label in enumerate(self.label):
            newlabel = n+1
            seg[data==label]=newlabel
        
        seg, _ = pad(seg, self.size)
        return seg

    def __getitem__(self, idx):    
        sub = self.movingsub[idx]
        image, _ = self.preprocess_img(f'{self.datapath}/{sub}/{sub[4:]}_{self.mod}.nii.gz')
        seg = self.preprocess_seg(f'{self.datapath}/{sub}/{sub[4:]}_seg.nii.gz')
        return self.fiximg, self.fixseg, self.fix_nopad, image, seg 


def BraTS_dataloader(args, root_path, size=[240, 240, 160]):
    train_data = BraTSDataset(f'{root_path}/BraTS2018_Train', args.mod, size=size)
    
    logging.info(f'Train:{len(train_data)}')

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bsize,
        shuffle=True,
        drop_last= True,
        num_workers=args.num_workers)
    
    if args.eval:
        test_data = BraTSDataset(f'{root_path}/BraTS2018_Validation', args.mod, size=size)
        logging.info(f'Test:{len(test_data)}')
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.bsize,
            shuffle=False,
            num_workers=args.num_workers)
    else:
        test_dataloader = None
    return train_dataloader, test_dataloader