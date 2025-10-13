import torch
import torch.nn as nn

from utils.util import EasyDict as edict
from utils.loss import Loss
from model.shape.implicit import Implicit
from model.shape.seen_coord_enc import CoordEncAtt, CoordEncRes
from model.shape.rgb_enc import RGBEncAtt, RGBEncRes
from model.depth.dpt_depth import DPTDepthModel
from model.shape.symm_decoder import SymmDecoder
from utils.util import toggle_grad, interpolate_coordmap, get_child_state_dict
from utils.camera import unproj_depth, valid_norm_fac
from utils.layers import Bottleneck_Conv
from utils.pos_embed import get_2d_sincos_pos_embed


import numpy as np
import json
from scipy.optimize import linear_sum_assignment

from transformers import AutoImageProcessor, AutoModel

class Graph(nn.Module):

    def __init__(self, opt):
        super().__init__()

        if opt.optim.use_vit:
            self.dino_model = AutoModel.from_pretrained('google/vit-large-patch16-224')
            self.feature_converter = nn.Sequential(
                nn.Linear(197, 257),
            )
        else:
            self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large')

        feat_res = opt.H // opt.arch.win_size
        
        # loss functions
        self.loss_fns = Loss(opt)

        # decoder for symmetry
        with open("data/symm/candidate_planes.json") as json_file:
            candidate_planes = json.load(json_file)

        self.query_normals = torch.from_numpy(np.array(candidate_planes)).float()
        self.symm_decoder = SymmDecoder(opt, self.query_normals)

        num_patches = feat_res**2
        n_channels = opt.arch.impl.n_channels
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_channels), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(feat_res), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def sample_directions_v2(self, l):
        phi = np.linspace(0, 2 * np.pi, 2*l+1)
        # costheta = np.linspace(0, 1, l+1)
        theta = np.linspace(0, np.pi, l+1)
        theta, phi = np.meshgrid(theta, phi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        directions = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        return directions

    def gen_candidate_planes(self, opt):
        gen_dir = self.sample_directions_v2(6)
        gen_dir = np.unique(gen_dir, axis=0)
        gen_dir_filtered = []
        for item in gen_dir:
            to_add = True
            for existing_item in gen_dir_filtered:
                if np.abs(np.dot(existing_item, item)) > 1-1e-6:
                    to_add = False
                    break
            if to_add:
                gen_dir_filtered.append(item)
        gen_dir_filtered_list = [x.tolist() for x in gen_dir_filtered]
        return gen_dir_filtered_list





    def load_pretrained_depth(self, opt):
        if opt.pretrain.depth:
            # loading from our pretrained depth and intr model
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.pretrain.depth))
            checkpoint = torch.load(opt.pretrain.depth, map_location="cuda:{}".format(opt.device))
            self.dpt_depth.load_state_dict(get_child_state_dict(checkpoint["graph"], "dpt_depth"))
            # load the intr head
            if opt.device == 0:
                print("loading pretrained intr from {}...".format(opt.pretrain.depth))
            self.intr_head.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_head"))
            self.intr_proj.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_proj"))
        elif opt.arch.depth.pretrained:
            # loading from omnidata weights
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.arch.depth.pretrained))
            checkpoint = torch.load(opt.arch.depth.pretrained, map_location="cuda:{}".format(opt.device))
            state_dict = checkpoint['model_state_dict']
            self.dpt_depth.load_state_dict(state_dict)

    def intr_param2mtx(self, opt, intr_params):
        '''
        Parameters:
            opt: config
            intr_params: [B, 3], [scale_f, delta_cx, delta_cy]
        Return:
            intr: [B, 3, 3]
        '''
        batch_size = len(intr_params)
        f = 1.3875
        intr = torch.zeros(3, 3).float().to(intr_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
        intr[:, 2, 2] += 1
        # scale the focal length
        # range: [-1, 1], symmetric
        scale_f = torch.tanh(intr_params[:, 0])
        # range: [1/4, 4], symmetric
        scale_f = torch.pow(4. , scale_f)
        intr[:, 0, 0] += f * opt.W * scale_f
        intr[:, 1, 1] += f * opt.H * scale_f
        # shift the optic center, (at most to the image border)
        shift_cx = torch.tanh(intr_params[:, 1]) * opt.W / 2
        shift_cy = torch.tanh(intr_params[:, 2]) * opt.H / 2
        intr[:, 0, 2] += opt.W / 2 + shift_cx
        intr[:, 1, 2] += opt.H / 2 + shift_cy
        return intr
    
    def prepare_symm_targets(self, opt, var):
        return {
            "gt_normal_normalized": var.gt_normal_cam,
            "center_coords": var.gt_center_cam,
            "num_actual_gt": var.num_actual_gt,
            "num_total_actual_gt": var.num_actual_gt.sum(),
        }
    
    def prepare_symm_output(self, opt, var, outputs):
        targets = var.symm_targets
        normal_dist = 1 - torch.matmul(outputs["normal_normalized"], targets["gt_normal_normalized"].transpose(1, 2))

        centers = targets["center_coords"]
        centers = centers.squeeze(1)
        # max_num_planes = outputs["normal_normalized"].shape[1]
        # centers = centers.unsqueeze(1).repeat(1, max_num_planes, 1)

        # squared distance to center
        # Ax+By+Cz+D=0, center: (x,y,z), normal: (A,B,C), D from outputs
        # Input: outputs["normal_normalized"]: (B, Q, 3), centers: (B, 3), outputs["plane_offset"]: (B, Q, 1)
        # Output: outputs["center_dist"]: (B, Q)
        center_dist = torch.matmul(outputs["normal_normalized"], centers.unsqueeze(-1)) + outputs["plane_offset"]
        center_dist = center_dist.squeeze(-1)
        center_dist = center_dist ** 2
    
        return {
            "cls_prob": outputs["cls_prob"],
            "cls_logits": outputs["cls_logits"], 
            "normal": outputs["normal_normalized"],
            "offset": outputs["plane_offset"],
            "center_dist": center_dist,
            "normal_dist": normal_dist
        }
    
    def prepare_symm_output_demo(self, opt, var, outputs):    
        return {
            "cls_prob": outputs["cls_prob"],
            "cls_logits": outputs["cls_logits"], 
            "normal": outputs["normal_normalized"],
            "offset": outputs["plane_offset"],
        }
    
    def prepare_symm_assignments(self, opt, var, query_normals):
        outputs = var.symm_outputs
        batch_size = outputs["cls_prob"].shape[0]
        nprop = outputs["cls_prob"].shape[1]
        device = outputs["cls_prob"].device

        targets = var.symm_targets
        num_actual_gt = targets["num_actual_gt"]
        gt_normal_normalized = targets["gt_normal_normalized"]

        query_normals = query_normals.to(device).unsqueeze(0).repeat(batch_size, 1, 1)

        normal_query_gt_dist = 1 - torch.matmul(query_normals, gt_normal_normalized.transpose(1, 2))
        final_cost = normal_query_gt_dist.detach().cpu().numpy()

        assignments = []

        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            assign = []
            if num_actual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : num_actual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }
    
    def visualize_gt_planes(self, var):
        import pyvista as pv
        import trimesh

        gt_points = var.gt_points_cam.squeeze(0).cpu().numpy()
        gt_center = var.symm_targets['center_coords'].squeeze(0).cpu().numpy().tolist()
        # TODO: add cpoint to visualization
        gt_normal = var.symm_targets['gt_normal_normalized'].squeeze(0).cpu().numpy().tolist()
        gt_num = var.symm_targets["num_actual_gt"].squeeze(0).cpu().numpy().tolist()[0]

        pred_center = var.symm_outputs["center_dist"].squeeze(0).cpu().numpy()


        plotter = pv.Plotter()
        plotter.show_axes()
        plotter.view_xy()
        # meshsphere = trimesh.creation.icosphere(radius=1)
        pvpc = pv.PolyData(gt_points)
        plotter.add_mesh(pvpc, color='blue')
        # plotter.add_mesh(pv.wrap(meshsphere), color='pink')
        for d in gt_normal[:gt_num]:
            plotter.add_mesh(pv.Plane(center=gt_center, direction=d, i_size=10, j_size=10), 
                            color='green')
        # for d in [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]:
        #     plotter.add_mesh(pv.Plane(center=[0.,0.,0.], direction=d, i_size=10, j_size=10), 
        #                     color='green')
        plotter.show()


    def forward(self, opt, var, training=False, get_loss=True, visualize=False):
        batch_size = len(var.idx)

        # B, 3, 224, 224
        if opt.optim.freeze_dino:
            with torch.no_grad():
                var.latent_depth = self.dino_model(var.rgb_input_map)['last_hidden_state'] # Bx257x1024
        else:
            var.latent_depth = self.dino_model(var.rgb_input_map)['last_hidden_state']

        if opt.optim.use_vit:
            var.latent_depth = self.feature_converter(var.latent_depth.transpose(1, 2)).transpose(1, 2)

        
        var.enc_pos = self.pos_embed
        var.pose = var.pose_gt
        # forward for loss calculation (only during training)
        if 'gt_sample_points' in var and 'gt_sample_sdf' in var:
            with torch.no_grad():
                # get the normalizing fac based on the GT seen surface
                # project the GT depth to 3D points in view-centric frame
                # [B, H*W, 3], in camera coordinates
                seen_points_3D_gt = unproj_depth(opt, var.depth_input_map, var.intr)
                # [B, H*W, 3], [B, 1, H, W] (boolean) -> [B, 3], [B]
                seen_points_mean_gt, seen_points_scale_gt = valid_norm_fac(seen_points_3D_gt, var.mask_input_map > 0.5)
                var.seen_points_gt = (seen_points_3D_gt - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                var.seen_points_gt[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
                
                # transform the GT points accordingly
                R_gt = var.pose_gt[:, :, :3]
                T_gt = var.pose_gt[:, :, 3:]
                gt_sample_points_transposed = var.gt_sample_points.permute(0, 2, 1).contiguous()
                gt_sample_points_cam = (R_gt @ gt_sample_points_transposed + T_gt).permute(0, 2, 1).contiguous()
                var.gt_points_cam = (gt_sample_points_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                
                gt_center_transposed = var.center_coords.permute(0, 2, 1).contiguous()
                gt_normal_transposed = var.gt_normal_normalized.permute(0, 2, 1).contiguous()
                # camera coordinates, [B, N, 3]
                gt_center_cam = (R_gt @ gt_center_transposed + T_gt).permute(0, 2, 1).contiguous()
                gt_normal_cam_dirty = (R_gt @ gt_normal_transposed + T_gt).permute(0, 2, 1).contiguous()
                # normalize with seen std and mean, [B, N, 3]
                var.gt_center_cam = (gt_center_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)

                gt_normal_cam_dirty = (gt_normal_cam_dirty - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                gt_normal_cam_dirty -= var.gt_center_cam
                gt_normal_cam_dirty = gt_normal_cam_dirty / torch.norm(gt_normal_cam_dirty, dim=-1, keepdim=True)
                bs = gt_normal_cam_dirty.size(0)
                for i in range(bs):
                    gt_normal_cam_dirty[i, var.num_actual_gt[i]:, :] = 0
                var.gt_normal_cam  = gt_normal_cam_dirty
                

            query_normals = self.query_normals.to(var.latent_depth.device)
            if opt.arch.depth.encoder == 'resnet':
                symm_outputs = self.symm_decoder(var.latent_depth, pre_computed_pos_emb=True, enc_pos=var.enc_pos)
            elif opt.arch.depth.encoder == 'pointtr':
                symm_outputs = self.symm_decoder(var.latent_depth, enc_xyz=var.enc_xyz)
            else:
                # throw not implemented error
                raise NotImplementedError("Symmetry decoder not implemented for this encoder")
            symm_outputs = symm_outputs["outputs"] # ommitted the "aux_outputs" 
            var.symm_targets = self.prepare_symm_targets(opt, var)
            var.symm_outputs = self.prepare_symm_output(opt, var, symm_outputs)
            var.symm_assignments = self.prepare_symm_assignments(opt, var, query_normals)
            if visualize:
                self.visualize_gt_planes(var)

        else:
            query_normals = self.query_normals.to(var.latent_depth.device)
            if opt.arch.depth.encoder == 'resnet':
                symm_outputs = self.symm_decoder(var.latent_depth, pre_computed_pos_emb=True, enc_pos=var.enc_pos)
            elif opt.arch.depth.encoder == 'pointtr':
                symm_outputs = self.symm_decoder(var.latent_depth, enc_xyz=var.enc_xyz)
            else:
                # throw not implemented error
                raise NotImplementedError("Symmetry decoder not implemented for this encoder")
            symm_outputs = symm_outputs["outputs"] # ommitted the "aux_outputs" 
            var.symm_outputs = self.prepare_symm_output_demo(opt, var, symm_outputs)
        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        
        return var

    
    def get_reflection_mx_via_center(self, symm_normal_used):
        reflection_mx = torch.zeros((symm_normal_used.shape[0], 3, 3), device=symm_normal_used.device)
        for b in range(symm_normal_used.shape[0]):
            reflection_mx[b] = torch.eye(3).to(symm_normal_used.device) - 2 * torch.outer(symm_normal_used[b], symm_normal_used[b])
            
        return reflection_mx # B x 3 x 3
    
    def reflect_query_points(self, reflection_mx, gt_symm_center_used, points):
        center = gt_symm_center_used
        points = points - center
        reflected_points = torch.bmm(reflection_mx, points.permute(0, 2, 1)).permute(0, 2, 1)
        reflected_points = reflected_points + center
        return reflected_points

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.symm_cls is not None and training:
            loss.symm_cls = self.loss_fns.symm_cls_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)
        if opt.loss_weight.symm_normal is not None and training:
            loss.symm_normal = self.loss_fns.symm_normal_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)
        if opt.loss_weight.symm_offset is not None and training:
            loss.symm_offset = self.loss_fns.symm_offset_loss(var.symm_outputs, var.symm_targets, var.symm_assignments)

        return loss
