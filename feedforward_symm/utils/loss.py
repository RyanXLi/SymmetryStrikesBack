import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from copy import deepcopy
from model.depth.midas_loss import MidasLoss

import torch.nn.functional as F
# from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class Loss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)
        # self.occ_loss = nn.BCEWithLogitsLoss(reduction='none')
        if opt.optim.amp:
            self.occ_loss = self.binary_cross_entropy
        else:
            self.occ_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.midas_loss = MidasLoss(alpha=opt.training.depth_loss.grad_reg, 
                                    inverse_depth=opt.training.depth_loss.depth_inv, 
                                    shrink_mask=opt.training.depth_loss.mask_shrink)
        # self.semcls_percls_weights = torch.ones(2)
        # self.semcls_percls_weights[0] = 0.1
        self.neg_class_weight = 0.1
        # self.symm_cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.symm_cls_loss_fn = FocalLoss(alpha=1, gamma=2, reduction='none')

    def binary_cross_entropy(self, my_input, target):
        """
        F.binary_cross_entropy is not numerically stable in mixed-precision training.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            my_input = my_input.float()
            target = target.float()
            my_input = torch.sigmoid(my_input)
            result = -(target * torch.log(my_input) + (1 - target) * torch.log(1 - my_input))
        return result
    
    def alignment_loss(self, my_input, target):
        return -(target * torch.log(my_input) + (1 - target) * torch.log(1 - my_input))


    def shape_loss(self, pred_occ_raw, gt_sdf):
        assert len(pred_occ_raw.shape) == 2
        assert len(gt_sdf.shape) == 2
        # [B, N]
        gt_occ = (gt_sdf < 0).float()
        loss = self.occ_loss(pred_occ_raw, gt_occ)
        weight_mask = torch.ones_like(loss)
        thres = self.opt.training.shape_loss.impt_thres
        weight_mask[torch.abs(gt_sdf) < thres] = weight_mask[torch.abs(gt_sdf) < thres] * self.opt.training.shape_loss.impt_weight 
        loss = loss * weight_mask
        return loss.mean()

    def depth_loss(self, pred_depth, gt_depth, mask):
        assert len(pred_depth.shape) == len(gt_depth.shape) == len(mask.shape) == 4
        assert pred_depth.shape[1] == gt_depth.shape[1] == mask.shape[1] == 1
        loss = self.midas_loss(pred_depth, gt_depth, mask)
        return loss
    
    def intr_loss(self, seen_pred, seen_gt, mask):
        assert len(seen_pred.shape) == len(seen_gt.shape) == 3
        assert len(mask.shape) == 2
        # [B, HW]
        distance = torch.sum((seen_pred - seen_gt)**2, dim=-1)
        loss = (distance * mask).sum() / (mask.sum() + 1.e-8)
        return loss
    
    def symm_cls_loss(self, outputs, targets, assignments):
        pred_logits = outputs["cls_logits"].squeeze(-1)
        gt_box_label = assignments["proposal_matched_mask"]

        loss = self.symm_cls_loss_fn(pred_logits, gt_box_label)
        weight_mask = torch.ones_like(loss)
        weight_mask[gt_box_label == 0] = self.neg_class_weight
        loss = loss * weight_mask
        return loss.mean()

        # loss = sigmoid_focal_loss(
        #     pred_logits,
        #     gt_box_label,
        #     # self.semcls_percls_weights,
        #     reduction="mean",
        # )


    
    def symm_normal_loss(self, outputs, targets, assignments):
        normal_dist = outputs["normal_dist"]
        

        # select appropriate distances by using proposal to gt matching
        # this leaves only the valid ones, and filters out all gt that is not matching
        normal_loss = torch.gather(
            normal_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        normal_loss = normal_loss * assignments["proposal_matched_mask"]
        normal_loss = normal_loss.sum()
        if targets["num_total_actual_gt"] > 0:
            normal_loss /= targets["num_total_actual_gt"]

        return normal_loss   

    def symm_offset_loss(self, outputs, targets, assignments):
        
        center_dist = outputs["center_dist"]
        
        # zero-out non-matched proposals
        offset_loss = center_dist * assignments["proposal_matched_mask"]
        offset_loss = offset_loss.sum()
        if targets["num_total_actual_gt"] > 0:
            offset_loss /= targets["num_total_actual_gt"]

        return offset_loss
    
    def alignment_loss_12(self, outputs, flipped_outputs, gt_sdf, batch_dimension_valid_mask):
        # Align predicted occ with pred flipped occ or gt flipped occ (loss 1 or loss 2)
        # In graph shape: extract pred and gt symmetry planes, get the flipping matrix by these planes, flip the query points, and then query model, and calculate the loss

        outputs = torch.sigmoid(outputs.float())
        flipped_outputs = torch.sigmoid(flipped_outputs.float())

        loss = self.alignment_loss(outputs, flipped_outputs)
        weight_mask = torch.ones_like(loss)
        thres = self.opt.training.shape_loss.impt_thres
        weight_mask[torch.abs(gt_sdf) < thres] = weight_mask[torch.abs(gt_sdf) < thres] * self.opt.training.shape_loss.impt_weight 
        loss = loss * weight_mask
        loss *= batch_dimension_valid_mask.unsqueeze(-1)
        return loss.mean()
    
    def alignment_loss_3(self, flipped_outputs, gt_sdf, batch_dimension_valid_mask):
        assert len(flipped_outputs.shape) == 2
        assert len(gt_sdf.shape) == 2
        # [B, N]
        gt_occ = (gt_sdf < 0).float()
        loss = self.occ_loss(flipped_outputs, gt_occ)
        weight_mask = torch.ones_like(loss)
        thres = self.opt.training.shape_loss.impt_thres
        weight_mask[torch.abs(gt_sdf) < thres] = weight_mask[torch.abs(gt_sdf) < thres] * self.opt.training.shape_loss.impt_weight 
        loss = loss * weight_mask
        loss *= batch_dimension_valid_mask.unsqueeze(-1)
        return loss.mean()
    
    def symmshape_loss(self, outputs, flipped_outputs, gt_sdf, batch_dimension_valid_mask):
        # outputs = torch.sigmoid(outputs.float())
        # flipped_outputs = torch.sigmoid(flipped_outputs.float())
        avg_outputs = 0.5 * (outputs + flipped_outputs)
        final_outputs = torch.where(batch_dimension_valid_mask.unsqueeze(-1), avg_outputs, outputs)

        gt_occ = (gt_sdf < 0).float()
        loss = self.occ_loss(final_outputs, gt_occ)
        weight_mask = torch.ones_like(loss)
        thres = self.opt.training.shape_loss.impt_thres
        weight_mask[torch.abs(gt_sdf) < thres] = weight_mask[torch.abs(gt_sdf) < thres] * self.opt.training.shape_loss.impt_weight 
        loss = loss * weight_mask
        return loss.mean()

