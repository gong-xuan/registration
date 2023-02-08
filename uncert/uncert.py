import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
# setting path
import sys
sys.path.append('../')
import losses

uncertainty_metrics = ['rmse']

def uncert_fromts(fix, warped, uncert, nopad=None):
    fix = fix.detach().cpu().numpy()
    warped = warped.detach().cpu().numpy()
    uncert = uncert.detach().cpu().numpy()
    nopad = nopad.detach().cpu().numpy()
    uncer = np.sqrt(uncert[:,:,:,0]**2+uncert[:,:,:,1]**2+uncert[:,:,:,2]**2)/3
    # import ipdb; ipdb.set_trace()
    r = compute_uncert_metrics(fix.squeeze(), warped.squeeze(), uncer)
    return r

def compute_errors(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    results = []
    
    if mask is not None:
        pred = pred[:,mask]
        gt = gt[:,mask]
    
    if "abs_rel" in metrics:
        abs_rel = (np.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = ((gt - pred) ** 2)
        if reduce_mean:
            # rmse = np.sqrt(rmse.mean())
            rmse = rmse.mean()
        else:
            rmse =rmse.mean(axis=0)
        results.append(rmse)

    if "a1" in metrics:
        a1 = np.maximum((gt / pred), (pred / gt))
        if reduce_mean:
            # invert to get outliers
            a1 = (a1 >= 1.25).mean()
        results.append(a1)

    return results

def compute_uncert_metrics(gt, pred, uncert, intervals = 50):
    """
        gt : fixed image
        pred : moving image after getting warped
        uncert : uncertainty estimates

    """
    # results dictionaries
    AUSE = { "rmse":0}
    AURG = { "rmse":0}
    # AUSE = {"abs_rel":0, "rmse":0, "a1":0}
    # AURG = {"abs_rel":0, "rmse":0, "a1":0}
    uncert = -uncert #flip signs so that the highest absolute value of uncertainty measurement is considered first
    true_uncert = compute_errors(gt,pred)

    #true_uncert = {"abs_rel":-true_uncert[0],"rmse":-true_uncert[1],"a1":-true_uncert[2]}
    true_uncert ={"rmse":-true_uncert[0]}
    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    plotx = [1./intervals*t for t in range(0,intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]
    sparse_curve = {m:[compute_errors(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # import ipdb; ipdb.set_trace()
    opt_thresholds = {m:[np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    opt_curve = {m:[compute_errors(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    rnd_curve = {m:[compute_errors(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    for m in uncertainty_metrics:
        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    # import ipdb; ipdb.set_trace()

    return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}

def ensemble_uncert(x):
    assert len(x.shape)==5
    if x.shape[1] == 3:
        # y = (x[:,0,:,:,:]**2 + x[:,1,:,:,:]**2 + x[:,2,:,:,:]**2).sqrt()/3.0
        y = x.mean(dim=1)

    else:
        y = x.mean(dim=1)
    return y

def error_mean(err, indices=None):
    if indices is not None:
        return err[indices].mean()
    else:
        return err.mean()

def ts_l1(pred, gt, mask=None, reduce_mean=False):
    err = (pred- gt).abs()
    # import ipdb; ipdb.set_trace()
    err = torch.mean(err,dim=1)
    # err = ensemble_uncert(err)
    if mask is not None:
        err = err[mask]
    if reduce_mean:
        err = err.mean()
    return err

def ts_ncc(pred, gt, mask=None, reduce_mean=False):
    ncc, _ = losses.mask_ncc(pred, gt, reduce_mean=False)
    # en_ncc = ensemble_uncert(1+ncc)
    ncc = torch.mean(1+ncc,dim=1)
    if mask is not None:
        # import ipdb; ipdb.set_trace()
        # use_mask = use_mask*mask
        ncc = ncc[mask]
    
    if reduce_mean:
        return ncc.mean()
    else:
        return ncc

def ts_bce(pred, gt, mask=None, reduce_mean=False):
    pred_clip = torch.clamp(pred, min=1e-5, max=1-1e-5)
    err = torch.nn.BCELoss(reduction='none')(pred_clip, gt)
    # import ipdb; ipdb.set_trace()
    err = torch.mean(err,dim=1)
    # err = ensemble_uncert(err)
    if mask is not None:
        err = err[mask]
    if reduce_mean:
        err = err.mean()
    return err

def ts_dice(pred, gt, mask=None, reduce_mean=True, labels=[1,2]):
    v1 = pred.round().detach().cpu().numpy()
    v2 = gt.detach().cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()
        assert len(mask.shape)==len(pred.shape)
        mask = mask[:,0,:,:,:]
    
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        # import ipdb; ipdb.set_trace()
        vol1 = v1[:,label,:,:,:]
        vol2 = v2[:,label,:,:,:]
        if mask is not None:
            vol1 = vol1[mask]
            vol2 = vol2[mask]
        top = 2 * np.sum(np.logical_and(vol1, vol2))
        bottom = np.sum(vol1) + np.sum(vol2)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom
  
    return dicem.mean()

def to_1d(pred,gt,uncert):
    # print(gt.sum(dim=1).unique(), pred.sum(dim=1).unique())
    no_pad = (gt.sum(dim=1)!=0)
    no_pad_1d = no_pad.reshape(-1)
    gt_label = torch.max(gt,dim=1)[1]
    gt_label_1d = gt_label.reshape(-1)[no_pad_1d]
    pred_label = torch.max(pred,dim=1)[1]
    pred_label_1d = pred_label.reshape(-1)[no_pad_1d]
    # uncert_1d = ensemble_uncert(uncert).reshape(-1)[no_pad_1d]
    c_uncert = []
    for l in range(uncert.shape[1]):
        c_uncert.append(uncert[:,l,:].reshape(-1)[no_pad_1d])
    c_uncert = torch.stack(c_uncert) #c*N
    return pred_label_1d,gt_label_1d,c_uncert

def dice_err(prelabel, gtlabel, mask=None,labels=[0], mode='MR', print=True): #if not in mask , will not count
    EPS = torch.finfo(torch.float32).eps
    dices = []
    if mask is not None:
        prelabel = prelabel[mask]
        gtlabel= gtlabel[mask]
    for label in labels:
        pred = (prelabel==label).float()
        gt = (gtlabel==label).float()
        # top = (pred*gt).sum()*2
        # bottom = pred.sum()+gt.sum()
        # dice = 1.0*top/max(EPS, bottom)
        #
        # precesion = 1.0*(pred*gt).sum()/max(EPS,pred.sum())
        # recall =1.0*(pred*gt).sum()/max(EPS,gt.sum())
        TP = (pred*gt).sum()
        FN = ((1-pred)*gt).sum()
        FP = ((1-gt)*pred).sum()
        MR = FN/max(FN+TP, EPS)
        FDR = FP/max(FP+TP, EPS) 
        dice = MR if mode == 'MR' else FDR
        # import ipdb; ipdb.set_trace()
        # print(TP.cpu.numpy(), FN, FP)
        dices.append(dice)
    dices=torch.stack(dices)
    # import ipdb; ipdb.set_trace()
    # print(dices.mean())
    return dices.mean()

def seg_uncert(pred, gt, uncert_raw, intervals = 500,  metrics=['acc','dice'], savedir=''):#1-acc, 1-dice
    # all to 1D
    # import ipdb; ipdb.set_trace()
    prelabel, gtlabel, uncert = to_1d(pred, gt, uncert_raw) #eliminate padding area
    pixel_err = (prelabel!=gtlabel).float()#smaller the better
    true_uncert = pixel_err
    
    quants = [1-t/intervals for t in range(0,intervals)]#[1,0.98,0.96,......,0.02]
    plotx = [1./intervals*t for t in range(0,intervals+1)]#[0,0.02,...,0.98,1]
    # get percentiles for sampling and corresponding subsets
    C, totalsize = uncert.shape[0], uncert.shape[1]
    AUSEs = {}
    AURGs = {}

    true_uncert_seq = torch.sort(true_uncert).indices 
    for metric in metrics:
        if metric=='acc':
            uncert_seq = torch.sort(uncert.mean(dim=0)).indices # small to large
            
            erravg = pixel_err.mean()
            sparse_curve = [pixel_err[uncert_seq[:int(q*totalsize)]].mean() for q in quants]+[0]
            opt_curve = [pixel_err[true_uncert_seq[:int(q*totalsize)]].mean() for q in quants]+[0]
        else:
            c = int(metric[-1])
            mode = metric[:-1]
            uncert_seq = torch.sort(uncert[c]).indices # small to large

            erravg = dice_err(prelabel, gtlabel, mode=mode, labels=[c], mask=None)
            sparse_curve = [dice_err(prelabel, gtlabel, mode=mode, labels=[c], mask=uncert_seq[:int(q*totalsize)]) for q in quants]+[0]
            opt_curve = [dice_err(prelabel, gtlabel, mode=mode, labels=[c], mask=true_uncert_seq[:int(q*totalsize)]) for q in quants]+[0]

        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        # import ipdb; ipdb.set_trace()
        rnd_curve = [erravg for t in range(intervals+1)]
        AURG = rnd_curve[0] - np.trapz(sparse_curve, x=plotx)

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE = np.trapz(sparse_curve, x=plotx) - np.trapz(opt_curve, x=plotx)
        
        AUSEs[metric] = AUSE.item()
        AURGs[metric] = AURG.item()

        if savedir:
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            fig, ax = plt.subplots(1,1)
            ax.plot(plotx, opt_curve, 'r', plotx, sparse_curve, 'g', plotx, rnd_curve, 'b')
            fig.savefig(os.path.join(savedir, f'{metric}.png'))
            #save array
            np.save(os.path.join(savedir,f'{metric}_opt.npy'), opt_curve)
            np.save(os.path.join(savedir,f'{metric}_sparse.npy'), sparse_curve)
            logging.info(f'saved....{metric}')
        
    # import ipdb; ipdb.set_trace()  
    return AUSEs, AURGs


def ts_uncert(pred, gt, uncert, mask=None, intervals = 500, metric='ncc', savemark='', savedir=''):
    pixel_error = ts_bce
    if metric=='ce':
        metric_error = ts_bce  
    elif metric=='ncc':
        metric_error = ts_ncc
    elif metric=='l1':
        metric_error = ts_l1
    # else:
    #     metric_error = ts_dice
    # import ipdb; ipdb.set_trace()
    uncert = ensemble_uncert(uncert) #the smaller, the better 
    # 
    # fig, ax = plt.subplots(4,2)
    # ax[0,0].imshow(uncert[0,0,:,:].cpu())
    # for i in range(3):
    #     ax[i+1,0].imshow(pred[0,i,23,:,:].cpu()) 
    #     ax[i+1,1].imshow(gt[0,i,23,:,:].cpu()) 
    # fig.savefig('love.png')
    # 
    pixel_err = metric_error(pred, gt, reduce_mean=False) #pixel
    true_uncert = pixel_err #the smaller, the better
    #for rank view all to 1D
    import ipdb; ipdb.set_trace()
    if mask is not None:
        uncert=uncert[mask]
        true_uncert=true_uncert[mask]
        pixel_err=pixel_err[mask]
    else:
        uncert = uncert.view(-1)
        true_uncert = true_uncert.view(-1)
        pixel_err = pixel_err.view(-1)
    # prepare subsets for sampling and for area computation
    # quants = [100./intervals*(intervals-t) for t in range(0,intervals)]
    quants = [1-t/intervals for t in range(0,intervals)]#[1,0.98,0.96,......,0.02]
    plotx = [1./intervals*t for t in range(0,intervals+1)]
    # get percentiles for sampling and corresponding subsets
    totalsize = uncert.shape[0]
    uncert_seq = torch.sort(uncert).indices
    true_uncert_seq = torch.sort(true_uncert).indices
    sparse_curve = [error_mean(pixel_err, indices=uncert_seq[:int(q*totalsize)]) for q in quants]+[0]
    # import ipdb; ipdb.set_trace()
    opt_curve = [error_mean(pixel_err, indices=true_uncert_seq[:int(q*totalsize)]) for q in quants]+[0]
    # rank_uncert = torch.sort(uncert)
    # uncert, uncert_indices = rank_uncert.values, rank_uncert.indices
    # rank_true_uncert = torch.sort(true_uncert)
    # true_uncert, true_uncert_indices = rank_true_uncert.values, rank_true_uncert.indices

    # sparse_curve = [error_mean(pixel_err, topsize=sub)for q in quants]+[0]
    # thresholds = [np.percentile(uncert.cpu(), q) for q in quants]
    # subs = [(uncert <= t) for t in thresholds]
    # sparse_curve = [error_mean(pixel_err, mask=sub)for sub in subs]+[0]

    # opt_thresholds = [np.percentile(true_uncert.cpu(), q) for q in quants]
    # opt_subs = [(true_uncert <= o) for o in opt_thresholds]
    # import ipdb; ipdb.set_trace()
    # opt_curve = [error_mean(pixel_err, mask=opt_sub)for opt_sub in opt_subs]+[0] 
    # import ipdb; ipdb.set_trace()
    erravg = error_mean(pixel_err, indices=None)
    rnd_curve = [erravg for t in range(intervals+1)]

    # import ipdb; ipdb.set_trace()
    # error: subtract from method sparsification (first term) the oracle sparsification (second term)
    AUSE = np.trapz(sparse_curve, x=plotx) - np.trapz(opt_curve, x=plotx)
    # gain: subtract from random sparsification (first term) the method sparsification (second term)
    AURG = rnd_curve[0] - np.trapz(sparse_curve, x=plotx)
    if savemark and savedir:
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        fig, ax = plt.subplots(1,1)
        ax.plot(plotx, opt_curve, 'r', plotx, sparse_curve, 'g', plotx, rnd_curve, 'b')
        fig.savefig(os.path.join(savedir, f'{savemark}.png'))
        
        #
        logging.info(f'saved....{savemark}')
    # import ipdb; ipdb.set_trace()  
    return AUSE.item(), AURG.item()


if __name__ == "__main__":
    size = torch.Size([3, 16,16,16])
    pred = torch.rand(size)
    gt = (pred>0.5).float()
    uncert = torch.rand(size).mean(dim=0)
    ause, aurg = ts_uncert(gt, pred, uncert)
    out = compute_uncert_metrics(gt.numpy(), pred.numpy(), uncert.numpy())
    import ipdb; ipdb.set_trace()

    # true_uncert = ts_error(pred, gt)

