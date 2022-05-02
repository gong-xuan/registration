import torch
import logging

def load_dict(savepath, model, dismiss_keywords=None): #dismiss_keywords is a list
    pth = torch.load(savepath)
    is_data_parallel = isinstance(model, torch.nn.DataParallel)#model
    new_pth = {}
    for k, v in pth.items():
        if dismiss_keywords is not None:
            exist = sum([dismiss_keywords[i] in k for i in range(len(dismiss_keywords))])
            if exist:
                continue
        if 'module' in k:
            if is_data_parallel: # saved multi-gpu, current multi-gpu
                new_pth[k] = v
            else: # saved multi-gpu, current 1-gpu 
                new_pth[k.replace('module.', '')] = v
        else: 
            if is_data_parallel: # saved 1-gpu, current multi-gpu
                new_pth['module.'+k] = v
            else: # saved 1-gpu, current 1-gpu
                new_pth[k] = v
    # import ipdb; ipdb.set_trace()
    m, u = model.load_state_dict(new_pth, strict=False)
    if m:
        logging.info('Missing: '+' '.join(m))
    if u:
        logging.info('Unexpected: '+' '.join(u))
    return

class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count