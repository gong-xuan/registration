
# setting path
import sys
sys.path.append('../')

import math
# from bisect import bisect_right,bisect_left
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class CyclicCosAnnealingLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from CosineAnnealingLR pytorch
    
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, epoch_per_cycle):
        # self.C = C
        # self.T = T
        self.epoch_per_cycle = epoch_per_cycle
        super(CyclicCosAnnealingLR, self).__init__(optimizer)

    def get_lr(self):
        # print(self.last_epoch)
        # return [base_lr for base_lr in self.base_lrs]
        # return [self.eta_min + (base_lr - self.eta_min) *
        #        (1 + math.cos(math.pi * curr_pos/ width)) / 2
        #         for base_lr in self.base_lrs]
        iteration = max(0, self.last_epoch -1)
        # import ipdb; ipdb.set_trace()
        # return [base_lr for base_lr in self.base_lrs]
        mod = iteration % self.epoch_per_cycle
        return [base_lr * (math.cos(math.pi * mod / self.epoch_per_cycle) + 1) / 2 for base_lr in self.base_lrs]



#################################
# TEST FOR SCHEDULER
#################################
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
net = nn.Sequential(nn.Linear(2,2))
# milestones = [(2**x)*300 for x in range(30)]
optimizer = optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005,nesterov=True)
scheduler = CyclicCosAnnealingLR(optimizer,50)
lr_log = []
for i in range(200):
    optimizer.step()
    scheduler.step()
    for param_group in optimizer.param_groups:
        lr_log.append(param_group['lr'])
        # print(i, param_group['lr'])

plt.plot(lr_log)
plt.show()
