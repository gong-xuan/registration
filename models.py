import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from losses import *
import logging
from utils import mk_grid_img, tensor2nii, flow_as_rgb
import matplotlib.pyplot as plt
import SimpleITK as sitk

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):#nearest??
        super(SpatialTransformer, self).__init__()
        self.zero = torch.cat([torch.eye(3), torch.zeros([3,1])], 1)[None]
        self.meshgrid = nn.Parameter(nnf.affine_grid(self.zero, [1,1,*size], align_corners=False), requires_grad=False)
        self.mode = mode

    def forward(self, src, flow, size=None):
        if size==None:
            meshgrid = self.meshgrid
        else:
            meshgrid = nn.Parameter(nnf.affine_grid(self.zero, [1,1,*size], align_corners=False), requires_grad=False).to(flow.device)
        flow = flow.permute(0, 2, 3, 4, 1)
        # print(meshgrid.shape, flow.shape)
        new_locs = meshgrid + flow
        self.new_locs = new_locs
        # import ipdb; ipdb.set_trace()
        return nnf.grid_sample(src, new_locs, mode=self.mode, align_corners=False)

class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 3
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim=3, enc_nf=[16, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16, 16], downsample=0, input_ch=2, return_enc=False):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_core, self).__init__()
        self.downsample = downsample
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = input_ch if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
            self.enc.append(nn.Dropout(0))
        # self.drop = nn.Dropout(droprate)

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5
        self.dec.append(conv_block(dim, dec_nf[4] + input_ch, dec_nf[5], 1)) # 6
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6]) 
        self.return_enc = return_enc
 

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        y = x
        for i in range(len(self.enc)):
            layer = self.enc[i]
            y = layer(y)
            # logging.info(y.size())
            if i%2==0:
                x_enc.append(y)
            # logging.info(layer)

        # save_enc output, for visualization if needed
        if self.return_enc:
           enc_out = y 
        
        # Three conv + upsample + concatenate series
        # y = x_enc[-1]
        # import ipdb; ipdb.set_trace()
        # y=self.drop(y)
        concat_cnt = 0
        upsample= 4-self.downsample
        upsample_cnt = 0
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.dec)-1):
            layer = self.dec[i]
            y = layer(y)
            # print(y.shape)
            if (i%2==0) and (upsample_cnt==upsample):#valid for up_cnt=0,1,2,3
                # import ipdb; ipdb.set_trace()
                return y
            if (i%2==0) and (concat_cnt<3):
                y = self.upsample(y)
                # logging.info(i, y.size(), x_enc[-(i+2)].size())
                y = torch.cat([y, x_enc[-(concat_cnt+2)]], dim=1) #(-2,-3,-4)
                concat_cnt += 1
                upsample_cnt += 1
                # print(upsample_cnt)
            # Two convs at full_size/2 res
            # y = layer(y)
            # y = self.dec[8](y)
            # y = self.dec[9](y)
            
        # Upsample to full res, concatenate and conv
        
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[-1](y)
        # Extra conv for vm2
        y = self.vm2_conv(y)
        if self.return_enc:
            return y, enc_out
        else:
            return y
    

class RegNet(nn.Module):
    def __init__(self, size, dim=3, winsize=7,
                enc_nf=[16, 32, 32, 32], 
                dec_nf= [32, 32, 32, 32, 32, 16, 16], 
                n_class=29,
                feat=False):
        super(RegNet, self).__init__()
        self.unet = unet_core(dim=dim, enc_nf = enc_nf, dec_nf = dec_nf, return_enc=True)
        if(dim == 3):
            conv_fn = nn.Conv3d
        else:
            conv_fn = nn.Conv2d
        self.conv = conv_fn(dec_nf[-1], dim, kernel_size = 3, stride = 1, padding = 1)
        self.spatial_transformer_network = SpatialTransformer(size)
        self.winsize = winsize
        self.n_class = n_class
        #feat
        self.feat = feat
        if self.feat:
            self.feat_conv= conv_fn(dec_nf[-1]+n_class, n_class, kernel_size = 3, stride = 1, padding = 1)
            self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout(droprate)


    def eval_dice_warped(self, fixed_label, warped_seg, seg_fname=None, save_nii=False):
        warplabel = torch.max(warped_seg.detach(),dim=1)[1]
        # print(torch.sum(warplabel))
        # import ipdb; ipdb.set_trace()
        warped_label = torch.nn.functional.one_hot(warplabel.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        # print(torch.sum(warped_label)) # Mahesh: This sum should reduce after taking one hot encoding, right?
        # print(torch.sum(fixed_label))
        # print(torch.sum(warped_label[:,1:,:,:,:]))
        # print(torch.sum(fixed_label[:,1:,:,:,:]))
        dice = dice_onehot(warped_label[:,1:,:,:,:].detach(), fixed_label[:,1:,:,:,:].detach())#disregard background
        
        if save_nii:
            fnames = []
            fnames.append(seg_fname+"true_seg")
            fnames.append(seg_fname+"pred_seg")
            tensor2nii(pred=warped_label, true=fixed_label, fnames=fnames, mode='seg', one_hot=True)
        # seg_fname = None # Temporary
        if seg_fname is not None:
            self.seg_imgs(warped_label, fixed_label, seg_fname, one_hot=True)
        return dice


    def seg_imgs(self, y_pred, y_true, fname, one_hot=False):
        if one_hot:
           y_pred = torch.max(y_pred.detach(), dim=1)[1]
           y_pred = y_pred.unsqueeze(0)
           y_true = torch.max(y_true.detach(), dim=1)[1]
           y_true = y_true.unsqueeze(0)
        fig_rows = 4
        fig_cols = 4
        n_subplots = fig_rows * fig_cols
        n_slice = y_true.shape[4]
        step_size = n_slice // n_subplots
        plot_range = n_subplots * step_size
        start_stop = int((n_slice - plot_range) / 2)
        fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
        y_true = np.array(y_true.detach().cpu().numpy()[0, 0, ...], dtype='float64')
        y_pred = np.array(y_pred.detach().cpu().numpy()[0, 0, ...], dtype='float64')
        y_true[ y_true==0 ] = np.nan
        for idx, img in enumerate(range(start_stop, plot_range, step_size)):
            axs.flat[idx].imshow(y_true[:, :, img])
            axs.flat[idx].imshow(y_pred[:, :, img], alpha=0.8)
            axs.flat[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(fname)
        plt.cla()
        plt.close()

    
    def forward(self, fix, moving, fix_label, moving_label, fix_nopad=None, rtloss=True, eval=True, seg_fname=None):
        x = torch.cat([moving, fix], dim = 1)
        unet_out, enc_out = self.unet(x)
        flow = self.conv(unet_out)
        # np.save('output2', flow.detach().cpu().numpy())
        
        if rtloss:
            warp = self.spatial_transformer_network(moving, flow)
            warped_seg = self.spatial_transformer_network(moving_label, flow)
            if self.feat:
                out_cat = torch.cat((warped_seg,unet_out), 1)
                out_cat = self.feat_conv(out_cat)
                warped_seg = self.softmax(out_cat)
            #loss
            if fix_nopad is not None:
                fix = fix*fix_nopad
                warp = warp*fix_nopad
            sim_loss, sim_mask = ncc_loss(warp, fix, reduce_mean=False, winsize=self.winsize) #[0,1]
            grad_loss = gradient_loss(flow, keepdim=False).mean()
            if fix_nopad is not None:
                mask = fix_nopad.bool()
                sim_mask = sim_mask*mask
            sloss = sim_loss[sim_mask].mean()
            ##seg_loss
            if fix_nopad is not None:
                fix_label = fix_label*fix_nopad
                warped_seg = warped_seg*fix_nopad
            warped_seg = torch.clamp(warped_seg, min=0, max=1)#required??
            seg_loss = tversky_loss(warped_seg, fix_label)
            
            # Mahesh : If similarity loss is greater than 80%, we will convert tensors to nifti for visualization.
            # if sloss*-1 >= 0.30:
            #     fnames = []
            #     fnames.append(seg_fname+"true.nii.gz")
            #     fnames.append(seg_fname+"warp.nii.gz")
            #     fnames.append(seg_fname+"flow.nii.gz")
            #     grid_flow = None
            #     # grid = mk_grid_img((400, 400, 64, 1, 3), 16, 1) 
            #     grid = mk_grid_img((256, 256, 64, 1, 3), 16, 1) # Mahesh : #TODO Somehow flow.shape doesnt work because 
            #     # max and min we get in flow_asrgb() both get value of 1. To overcome, passing hardcoded temporarily,
            #     # the axis along which max() and min() is taken will fix the issue.
            #     grid = np.moveaxis(grid, [3, 4], [-5, -4])
            #     grid = torch.tensor(grid, device=flow.device, dtype=flow.dtype)
            #     grid_flow = self.spatial_transformer_network(grid, flow)
            #     grid_flow = grid_flow.squeeze(0)
            #     if grid_flow is not None:
            #         fnames.append(seg_fname+"grid_flow.png")
            #     flow_as_rgb(grid_flow.detach().cpu().numpy(), 8, fname=fnames[-1])
            #     tensor2nii(warp, fix, fnames, mode='volume', one_hot=True, flow=flow, grid_flow=None)
            #     save_nii = True
            
            if eval:
                dice = self.eval_dice_warped(fix_label, warped_seg)
                return sloss, grad_loss, seg_loss, dice
            else:
                return sloss, grad_loss, seg_loss
        else:
            if eval:
                warped_seg= self.spatial_transformer_network(moving_label, flow)
                # print(torch.sum(warped_seg))
                # print(torch.sum(torch.max(warped_seg.detach(),dim=1)[1]))
                # print(torch.sum(fix_label))
                # print(torch.sum(torch.max(fix_label.detach(),dim=1)[1]))
                if self.feat:
                    out_cat = torch.cat((warped_seg, unet_out), 1)
                    out_cat = self.feat_conv(out_cat)
                    warped_seg = self.softmax(out_cat)
                # print(torch.sum(warped_seg))
                # print(torch.sum(torch.max(warped_seg.detach(),dim=1)[1]))
                if fix_nopad is not None:
                    fix_label = fix_nopad*fix_label
                    warped_seg = fix_nopad*warped_seg
                # print(torch.sum(warped_seg))
                # print(torch.sum(torch.max(warped_seg.detach(),dim=1)[1]))
                # print(torch.sum(fix_label))
                # print(torch.sum(torch.max(fix_label.detach(),dim=1)[1]))
                save_nii = False
                if seg_fname:
                    fnames = []
                    fnames.append(seg_fname+"true.nii.gz")
                    fnames.append(seg_fname+"warp.nii.gz")
                    fnames.append(seg_fname+"flow.nii.gz")
                    grid_flow = None
                    # grid_shape = (flow.shape[2], flow.shape[3], flow.shape[4], flow.shape[0], flow.shape[1])
                    # grid = mk_grid_img(grid_shape, 16, 1)
                    grid = mk_grid_img((240, 240, 96, 1, 3), 4, 1) # Mahesh : #TODO Somehow flow.shape doesnt work because 
                    # max and min we get in flow_as_rgb() both get value of 1. To overcome, passing hardcoded temporarily,
                    # the axis along which max() and min() is taken will fix the issue. But it still works if you set it some arbitary value, only
                    # used for flow images.
                    grid = np.moveaxis(grid, [3, 4], [-5, -4])
                    grid = torch.tensor(grid, device=flow.device, dtype=flow.dtype)
                    grid_flow = self.spatial_transformer_network(grid, flow)
                    grid_flow = grid_flow.squeeze(0)
                    if grid_flow is not None:
                        fnames.append(seg_fname+"grid_flow.png")
                    warp = self.spatial_transformer_network(moving, flow)
                    if fix_nopad is not None:
                        fix = fix*fix_nopad
                        warp = warp*fix_nopad
                    flow_as_rgb(grid_flow.detach().cpu().numpy(), 8, fname=fnames[-1])
                    tensor2nii(warp, fix, fnames, mode='volume', one_hot=True, flow=flow, grid_flow=None)
                    
                    # save the enc_output
                    enc_output = sitk.GetImageFromArray(enc_out.detach().cpu().numpy())
                    sitk.WriteImage(enc_output, seg_fname+"enc_out.nii.gz")
                    
                    save_nii = True
                dice = self.eval_dice_warped(fix_label, warped_seg, seg_fname=seg_fname, save_nii=save_nii)
                return dice
            else:
                return flow

"""
class RegUncertNet(RegNet):
    def __init__(self, size, dim=3, winsize=7,
                enc_nf=[16, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16, 16], n_class=29, grad_weight=0):
        super(RegNet, self).__init__()
        self.unet = unet_core(dim=dim, enc_nf = enc_nf, dec_nf = dec_nf)
        if(dim == 3):
            conv_fn = nn.Conv3d
        else:
            conv_fn = nn.Conv2d
        self.conv = conv_fn(dec_nf[-1], dim, kernel_size = 3, stride = 1, padding = 1)
        self.spatial_transformer_network = SpatialTransformer(size)
        self.winsize = winsize
        self.n_class = n_class
        #var
        self.conv_var = conv_fn(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv_var_2 = conv_fn(32, n_class, kernel_size = 3, stride = 1, padding = 1)
        self.softplus = nn.Softplus()
        self.grad_weight = grad_weight
        #feat
        # self.feat = feat
        # self.feat_conv= conv_fn(dec_nf[-1]+num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout(droprate)
    
    def forward(self, fix, moving, fix_label, moving_label, gtflow=None, rtloss=True, eval=True):
        x = torch.cat([moving,fix], dim = 1)
        unet_out = self.unet(x)
        flow = self.conv(unet_out)
        out = self.conv_var(unet_out)
        out = self.conv_var_2(out)
        flow_var = self.softplus(out)

        warp = self.spatial_transformer_network(moving, flow)
        #
        if rtloss:
            #var
            
            dis = torch.nn.L1Loss(reduction='none')(flow, gtflow).mean(dim=1, keepdim=True)
            loss = (dis/flow_var+ flow_var.log()).mean()
            if self.grad_weight>0:
                loss += gradient_loss(flow, keepdim=False).mean()
            # sim_loss, sim_mask = ncc_loss(warp,fix, reduce_mean=False, winsize=self.winsize) #[0,1]
            # grad_loss = gradient_loss(flow, keepdim=False).mean()
            # sloss = sim_loss[sim_mask].mean()
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow)
                return loss, dice
            else:
                return loss
        else:
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow)
                return dice
            else:
                warped_seg= self.spatial_transformer_network(moving_label, flow)
                warplabel = torch.max(warped_seg.detach(),dim=1)[1]
                warpseg = torch.nn.functional.one_hot(warplabel.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
                return warpseg, flow, flow_var
"""