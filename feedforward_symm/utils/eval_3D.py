import numpy as np
import torch
import threading
import mcubes
import trimesh
from external.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.util_vis import show_att_on_image
from utils.camera import get_rotation_sphere
import trimesh.repair

THRESHOLD = 0.5
THRESHOLD_FLIP = 0.8
ERR = 0.001

@torch.no_grad()
def get_dense_3D_grid(opt, var, N=None):
    batch_size = len(var.idx)
    N = N or opt.eval.vox_res
    # -0.6, 0.6
    range_min, range_max = opt.eval.range
    grid = torch.linspace(range_min, range_max, N+1, device=opt.device)
    points_3D = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij'), dim=-1) # [N, N, N, 3]
    # actually N+1 instead of N
    points_3D = points_3D.repeat(batch_size, 1, 1, 1, 1) # [B, N, N, N, 3]
    return points_3D

@torch.no_grad()
def compute_level_grid(opt, impl_network, latent_depth, latent_semantic, points_3D, images, vis_attn=False, pos=None):
    # needed for amp
    latent_depth = latent_depth.to(torch.float32) if latent_depth is not None else None
    latent_semantic = latent_semantic.to(torch.float32) if latent_semantic is not None else None
    
    # process points in sliced way
    batch_size = points_3D.shape[0]
    N = points_3D.shape[1]
    assert N == points_3D.shape[2] == points_3D.shape[3]
    assert points_3D.shape[4] == 3
    
    points_3D = points_3D.view(batch_size, N, N*N, 3)
    occ = []
    attn = []
    for i in range(N):
        # [B, N*N, 3]
        points_slice = points_3D[:, i]
        # [B, N*N, 3] -> [B, N*N], [B, N*N, 1+feat_res**2]
        occ_slice, attn_slice = impl_network(latent_depth, latent_semantic, points_slice, pos=pos)
        occ.append(occ_slice)
        attn.append(attn_slice.detach())
    # [B, N, N*N] -> [B, N, N, N]
    occ = torch.stack(occ, dim=1).view(batch_size, N, N, N)
    occ = torch.sigmoid(occ)
    return occ, None

@torch.no_grad()
def standardize_pc(pc):
    assert len(pc.shape) == 3
    pc_mean = pc.mean(dim=1, keepdim=True) 
    pc_zmean = pc - pc_mean
    origin_distance = (pc_zmean**2).sum(dim=2, keepdim=True).sqrt()
    scale = torch.sqrt(torch.sum(origin_distance**2, dim=1, keepdim=True) / pc.shape[1])
    pc_standardized = pc_zmean / (scale * 2)
    return pc_standardized

@torch.no_grad()
def normalize_pc(pc):
    assert len(pc.shape) == 3
    pc_mean = pc.mean(dim=1, keepdim=True) 
    pc_zmean = pc - pc_mean
    length_x = pc_zmean[:, :, 0].max(dim=-1)[0] - pc_zmean[:, :, 0].min(dim=-1)[0]
    length_y = pc_zmean[:, :, 1].max(dim=-1)[0] - pc_zmean[:, :, 1].min(dim=-1)[0]
    length_max = torch.stack([length_x, length_y], dim=-1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    pc_normalized = pc_zmean / (length_max + 1.e-7)
    return pc_normalized


def get_reflection_mx_via_center(normal):
    reflection_mx = torch.zeros((normal.shape[0], 3, 3), device=normal.device)
    for b in range(normal.shape[0]):
        reflection_mx[b] = torch.eye(3).to(normal.device) - 2 * torch.outer(normal[b], normal[b])
        
    return reflection_mx # B x 3 x 3

def reflect_query_points(reflection_mx, gt_symm_center_used, points):
    center = gt_symm_center_used
    points = points - center
    reflected_points = torch.bmm(reflection_mx, points.permute(0, 2, 1)).permute(0, 2, 1)
    reflected_points = reflected_points + center
    return reflected_points

@torch.no_grad()
def flip_level_grid(opt, var, points_3D):
    batch_size = points_3D.shape[0]
    orig_shape = points_3D.shape
    points_3D = points_3D.view(batch_size, -1, 3).contiguous()

    valid_mask = torch.zeros((batch_size), device=var.gt_sample_sdf.device).int()

    gt_planes = var.symm_targets["gt_normal_normalized"]
    num_actual_gt = var.symm_targets["num_actual_gt"]
    gt_symm_normal_used = torch.zeros((gt_planes.shape[0], 3), device=gt_planes.device)
    gt_symm_center_used = var.symm_targets["center_coords"]

    # randomly select one of the matched gt for each data in the batch
    for b in range(gt_planes.shape[0]):
        if num_actual_gt[b] == 0:
            gt_symm_normal_used[b, 0] = 1 # dummy value
            valid_mask[b] = 0
        else:
            # randomly select one of the gt for each data in the batch
            selected_gt = torch.randint(0, num_actual_gt[b], (1,))
            gt_symm_normal_used[b] = gt_planes[b, selected_gt]
            valid_mask[b] = 1

    valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).bool()
        
    gt_reflection_mx = get_reflection_mx_via_center(gt_symm_normal_used)
    flipped_points_3D_dirty = reflect_query_points(gt_reflection_mx, gt_symm_center_used, points_3D)
    flipped_points_3D = torch.where(valid_mask, flipped_points_3D_dirty, points_3D)
    flipped_points_3D = flipped_points_3D.view(*orig_shape).contiguous()

    return flipped_points_3D

def get_normal_cpoint_mask(opt, var, use_gt=False):
    batch_size = var.symm_targets["num_actual_gt"].shape[0]
    valid_mask = torch.zeros((batch_size), device=var.gt_sample_sdf.device).int()
    if use_gt:
        gt_planes = var.symm_targets["gt_normal_normalized"]
        num_actual_gt = var.symm_targets["num_actual_gt"]
        gt_symm_normal_used = torch.zeros((gt_planes.shape[0], 3), device=gt_planes.device)
        gt_symm_center_used = var.symm_targets["center_coords"]

        # randomly select one of the matched gt for each data in the batch
        for b in range(gt_planes.shape[0]):
            if num_actual_gt[b] == 0:
                gt_symm_normal_used[b, 0] = 1 # dummy value
                valid_mask[b] = 0
            else:
                # randomly select one of the gt for each data in the batch
                selected_gt = torch.randint(0, num_actual_gt[b], (1,))
                gt_symm_normal_used[b] = gt_planes[b, selected_gt]
                valid_mask[b] = 1

        valid_mask = valid_mask.bool()
        return gt_symm_normal_used, gt_symm_center_used, valid_mask
    else:
        # if there is prediction
        pred_logits = var.symm_outputs["cls_logits"] # B x 31
        pred_probs = torch.sigmoid(pred_logits)
        # find the max and index of the max for each data in the batch
        pred_confs, pred_ids = torch.max(pred_probs, dim=1)
        pred_symm_normal_used = torch.zeros((batch_size, 3), device=pred_logits.device)
        pred_symm_center_used = torch.zeros((batch_size, 3), device=pred_logits.device)

        for b in range(batch_size):
            if pred_confs[b] < THRESHOLD_FLIP:
                pred_symm_normal_used[b, 0] = 1 # dummy value
                valid_mask[b] = 0
            else:
                # randomly select one of the pred for each data in the batch
                selected_pred = pred_ids[b]
                pred_symm_normal_used[b] = var.symm_outputs["normal"][b, selected_pred]
                valid_mask[b] = 1
        valid_mask = valid_mask.bool()
        return pred_symm_normal_used, pred_symm_center_used, valid_mask




def flip_mesh(mesh, plane_normal, plane_point):
    # Normalize the plane normal vector
    plane_normal /= np.linalg.norm(plane_normal)
    
    # flip each vertex in the mesh
    vs = np.array(mesh.vertices)
    vs -= plane_point
    reflection_mx = np.eye(3) - 2 * np.outer(plane_normal, plane_normal)
    reflected_vs = np.dot(reflection_mx, vs.T).T
    reflected_vs += plane_point

    # Create a new mesh with the fliped vertices
    fliped_mesh = trimesh.Trimesh(vertices=reflected_vs, faces=mesh.faces)
    return fliped_mesh

def get_center_normal_from_2pc(pc1, pc2, N=10000):
    diff = pc1 - pc2
    # select N elements from diff with the largest norm
    _, indices = torch.topk(torch.norm(diff, dim=-1), N)
    diff = diff[indices]
    # choose the element with the largest norm as reference
    ref_idx = torch.norm(diff, dim=-1).argmax()
    diff = diff / torch.norm(diff, dim=-1, keepdim=True)
    ref = diff[ref_idx]
    # Adjust signs
    should_flip_sign = diff @ ref.unsqueeze(-1) < 0
    diff_adjusted = torch.where(should_flip_sign, -diff, diff)
    normal = diff_adjusted.mean(dim=0)
    # print(normal)
    # print(torch.norm(normal))
    new_center = (pc1.mean(dim=0) + pc2.mean(dim=0)) / 2
    return new_center, normal / torch.norm(normal)

def get_reflection_mx(normal):
    reflection_mx = torch.zeros((normal.shape[0], 3, 3), device=normal.device)
    for b in range(normal.shape[0]):
        reflection_mx[b] = torch.eye(3).to(normal.device) - 2 * torch.outer(normal[b], normal[b])
    return reflection_mx # B x 3 x 3

def reflect_pc(pc, center, normal): # [B, N, 3], [B, 3], [B, 3]
    pc_flip = pc - center.unsqueeze(1)
    pc_flip = pc_flip.float()
    reflection_mx = get_reflection_mx(normal)
    reflection_mx = reflection_mx.float().to(pc_flip.device)
    pc_flip = torch.bmm(reflection_mx, pc_flip.permute(0, 2, 1)).permute(0, 2, 1)
    pc_flip = pc_flip + center.unsqueeze(1)
    return pc_flip

def register_pc_and_mesh(opt, var, meshes, pointclouds):
    if opt.eval.flip_mesh:
        # TODO: only works with bs=1
        normals, cpoints, valid_masks = get_normal_cpoint_mask(opt, var, use_gt=opt.eval.flip_with_gt)
        if valid_masks[0] == 0:
            # not symmetric
            return torch.tensor(pointclouds, dtype=torch.float32, device=opt.device), meshes
        new_meshes = []
        new_pointclouds = []
        i = 0
        mesh = meshes[i]
        
        center = cpoints[i].cpu().numpy()
        normal = normals[i].cpu().numpy()
        flipped_mesh = flip_mesh(mesh, normal, center)
        
        if len(flipped_mesh.triangles)!=0:
            points = flipped_mesh.sample(opt.eval.num_points)
        else: points = np.zeros([opt.eval.num_points, 3])
        # new_meshes.append(flipped_mesh)
        new_pointclouds.append(points)
        # meshes = new_meshes
        # pointclouds = new_pointclouds

        pc_orig = torch.tensor(pointclouds, dtype=torch.float32, device=opt.device)
        pc_flipped = reflect_pc(pc_orig, cpoints, normals)
        # pc_flipped = torch.tensor(new_pointclouds, dtype=torch.float32, device=opt.device)

        pc_flipped_icp = ICP(opt, pc_flipped, pc_orig)

        final_pointclouds = []
        final_meshes = []

        

        for b in range(len(meshes)):
            if not mesh.is_watertight:
                # print the name of the data
                print(f"Mesh is not watertight!")
                return torch.tensor(pointclouds, dtype=torch.float32, device=opt.device), meshes
                result_mesh = mesh
            else:
                # print(pc_orig[b].shape) # 10000 x 3
                # print(pc_flipped_icp[b].shape)
                # exit(0)
                final_center, final_normal = get_center_normal_from_2pc(pc_orig[b], pc_flipped_icp[b], N=5000)
                final_center = final_center.cpu().numpy()
                final_normal = final_normal.cpu().numpy()

                # Create a plane box mesh, large enough to intersect your mesh entirely
                plane_extent = 4 * np.max(np.linalg.norm(mesh.vertices, axis=1))  

                normal_used = final_normal if np.dot(final_normal, np.array([0,0,1])) > 0 else -final_normal
                # Create a rotation matrix to align the z-axis with the plane normal
                rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], normal_used)
                # Create a translation matrix to move the plane to the point on the plane
                translation_matrix = trimesh.transformations.translation_matrix(final_center + normal_used * plane_extent / 2 + normal_used * ERR)
                # Combine the rotation and translation into a single transformation matrix
                transformation_matrix = np.dot(translation_matrix, rotation_matrix)

                plane_mesh = trimesh.creation.box(
                    extents=[plane_extent, plane_extent, plane_extent], 
                    transform=transformation_matrix)
                
                good_half_mesh = trimesh.boolean.difference([mesh, plane_mesh], engine='manifold')
                flipped_half_mesh = flip_mesh(good_half_mesh, final_normal, final_center)

                fixed_meshes = []
                for bad_mesh in [good_half_mesh, flipped_half_mesh]:
                    # Make winding consistent
                    trimesh.repair.fix_inversion(bad_mesh)
                    # Ensure normals are outward facing
                    trimesh.repair.fix_normals(bad_mesh)
                    fixed_meshes.append(bad_mesh)

                for f_mesh in fixed_meshes:
                    if not f_mesh.is_watertight:
                        # print the name of the data
                        print(f"Mesh is not watertight after cutting!")
                        return torch.tensor(pointclouds, dtype=torch.float32, device=opt.device), meshes

                try:
                    result_mesh = trimesh.boolean.union(fixed_meshes, engine='manifold')
                except:
                    print("Union failed")
                    return torch.tensor(pointclouds, dtype=torch.float32, device=opt.device), meshes
                
                if len(result_mesh.triangles)!=0:
                    points = result_mesh.sample(opt.eval.num_points)
                else: points = np.zeros([opt.eval.num_points, 3])
                final_pointclouds.append(points)
                final_meshes.append(result_mesh)

        final_pointclouds = torch.tensor(final_pointclouds, dtype=torch.float32, device=opt.device)
        return final_pointclouds, final_meshes        

    else:
        return torch.tensor(pointclouds, dtype=torch.float32, device=opt.device), meshes


def get_all_normal_cpoint_mask_from_pred(opt, var):
    prob_threshold = 0.1
    num_queries = opt.symm.num_queries
    pred_logits = var.symm_outputs["cls_logits"][0] # B x 31
    pred_probs = torch.sigmoid(pred_logits)

    valid_mask = torch.zeros((num_queries), device=var.gt_sample_sdf.device).int()
    pred_symm_normal_used = torch.zeros((num_queries, 3), device=pred_logits.device)
    pred_symm_center_used = torch.zeros((num_queries, 3), device=pred_logits.device)

    for i in range(pred_probs.shape[0]):
        if pred_probs[i] < prob_threshold:
            pred_symm_normal_used[i, 0] = 1 # dummy value
            valid_mask[i] = 0
        else:
            pred_symm_normal_used[i] = var.symm_outputs["normal"][0, i]
            valid_mask[i] = 1
    valid_mask = valid_mask.bool()
    return pred_symm_normal_used, pred_symm_center_used, valid_mask, pred_probs


def get_refined_dicts(opt, var, impl_network):
    results = []

    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox, _ = compute_level_grid(opt, impl_network, var.latent_depth, var.latent_semantic, 
                                                points_3D, var.rgb_input_map, vis_attn=False, pos=var.enc_pos)

    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # level_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes, pointclouds = convert_to_explicit(opt,level_grids,isoval=0.5,to_pointcloud=True)

    normals, cpoints, valid_masks, confs = get_all_normal_cpoint_mask_from_pred(opt, var)

    pc_orig = torch.tensor(pointclouds, dtype=torch.float32, device=opt.device)
    pc_orig = pc_orig.repeat(len(valid_masks), 1, 1)

    pc_flipped = reflect_pc(pc_orig, cpoints, normals)
    pc_flipped_icp, cd = ICP(opt, pc_flipped, pc_orig, return_cd=True)

    for i in range(len(valid_masks)):
        if valid_masks[i] == 0:
            results.append({})
            continue
        else:            
            final_center, final_normal = get_center_normal_from_2pc(pc_orig[i], pc_flipped_icp[i], N=5000)
            final_offsets = -torch.dot(final_center, final_normal)
            results.append({
                "normal": final_normal.cpu().numpy().tolist(), 
                "offset": final_offsets.cpu().numpy().tolist(),
                "icp_cd": cd.cpu().numpy().tolist(),
                "conf": confs[i].cpu().numpy().tolist()})
    return results

def parse_refined_results(opt, refined_results):
    results = {
            "refined_normal": [],
            "refined_offset": [],
            "refined_cd": [],
            "refined_conf": []
        }
    for i in range(len(refined_results)):
        if len(refined_results[i]) == 0:
            continue
        else:
            results["refined_normal"].append(refined_results[i]["normal"])
            results["refined_offset"].append(refined_results[i]["offset"])
            results["refined_cd"].append(refined_results[i]["icp_cd"])
            results["refined_conf"].append(refined_results[i]["conf"])
    return results


def eval_how_symm_is_shape(opt, var, meshes, pointclouds):
    return 0 # TODO: implement this


@torch.no_grad()
def eval_metrics_default(opt, var, impl_network, vis_only=False):
    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox, attn_vis = compute_level_grid(opt, impl_network, var.latent_depth, var.latent_semantic, 
                                             points_3D, var.rgb_input_map, vis_only, pos=var.enc_pos)
    
    if attn_vis:
        var.attn_vis = attn_vis
    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # level_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes,pointclouds = convert_to_explicit(opt,level_grids,isoval=0.5,to_pointcloud=True)

    if opt.eval.eval_how_symm_is_shape:
        shape_cd = eval_how_symm_is_shape(opt, var, meshes, pointclouds)
        

    var.dpc_pred, var.mesh_pred = register_pc_and_mesh(opt, var, meshes, pointclouds)

    # transform the gt to view-centered frame
    R_gt = var.pose_gt[..., :3]
    var.dpc.points = (R_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

    if opt.data.dataset_test == 'pix3d':
        var.dpc.points[:, :, :2] *= -1

    var.dpc_pred = normalize_pc(var.dpc_pred)
    var.dpc.points = normalize_pc(var.dpc.points)

    if vis_only: return
    if opt.eval.icp:
        # var.dpc_pred = ICP(opt, var.dpc_pred, var.dpc.points)
        np_pred = var.dpc_pred.cpu().numpy()
        np_gt = var.dpc.points.cpu().numpy()
        new_pred = []
        for i in range(batch_size):
            mx, new_np_pred, cost = trimesh.registration.icp(np_pred[i], np_gt[i])
            new_pred.append(new_np_pred)
        var.dpc_pred = torch.tensor(np.array(new_pred), dtype=torch.float32, device=opt.device)

    dist_acc, dist_comp, _, _ = chamfer_distance(opt, X1=var.dpc_pred, X2=var.dpc.points)
    var.f_score = compute_fscore(dist_acc, dist_comp, opt.eval.f_thresholds)   # [B, n_threshold]

    # var.symm_f1, var.symm_center_dist, var.symm_p, var.symm_r, var.symm_geo_dist = compute_symm_metrics(opt, var)
    # dist_acc: [B, n_points_pred]
    # dist_comp: [B, n_points_gt]
    assert dist_acc.shape[1] == opt.eval.num_points
    var.cd_acc = dist_acc.mean(dim=1)
    var.cd_comp = dist_comp.mean(dim=1)
    # var.shape_cd = shape_cd.mean(dim=1)

    if opt.eval.eval_how_symm_is_shape:
        return var.cd_acc.mean(), var.cd_comp.mean(), shape_cd.mean()
    return dist_acc.mean(), dist_comp.mean()

def brute_force_search(pc_pred, pc_gt, f_thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], device="cuda"):
    pc_pred = pc_pred.to(device).unsqueeze(0).float()
    pc_gt = pc_gt.to(device).unsqueeze(0).float().contiguous()
    pc_gt = normalize_pc(pc_gt)
    
    # brute force evaluation
    # get the best CD by iterating over the pose
    best_cd = np.inf
    rotations = get_rotation_sphere(azim_sample=24, elev_sample=24, roll_sample=12, scales=[1.0])
    batch_size = 24
    
    # process the pointcloud in batches
    for i in range(0, len(rotations), batch_size):
        rotation_batch = rotations[i:i+batch_size].to(device)
        pc_pred_rotated = (rotation_batch @ pc_pred.repeat(rotation_batch.shape[0], 1, 1).permute(0, 2, 1)).permute(0, 2, 1)
        pc_pred_rotated = normalize_pc(pc_pred_rotated).contiguous()
        acc, comp, _, _ = chamfer_distance(None, pc_pred_rotated, pc_gt.repeat(rotation_batch.shape[0], 1, 1).contiguous())
        f_score = compute_fscore(acc, comp, f_thresholds)
        acc, comp = acc.mean(dim=1), comp.mean(dim=1)
        cd = (acc + comp) / 2
        # save the best cd
        for j in range(len(cd)):
            if cd[j] < best_cd:
                best_pc_pred = pc_pred_rotated[j].clone()
                best_acc = acc[j]
                best_comp = comp[j]
                best_cd = cd[j]
                best_fscore = f_score[j]
                best_rotation = rotation_batch[j].clone()

    return best_acc, best_comp, best_fscore, best_pc_pred, pc_gt

def eval_metrics_BF(opt, var, impl_network, vis_only=False):
    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox, attn_vis = compute_level_grid(opt, impl_network, var.latent_depth, var.latent_semantic, 
                                             points_3D, var.rgb_input_map, vis_only, pos=var.enc_pos)
    if attn_vis:
        var.attn_vis = attn_vis
    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # level_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes, pointclouds = convert_to_explicit(opt, level_grids, isoval=0.5, to_pointcloud=True)
    var.dpc_pred, var.mesh_pred = register_pc_and_mesh(opt, var, meshes, pointclouds)

    # transform the gt to view-centered frame
    R_gt = var.pose_gt[..., :3]
    var.dpc.points = (R_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    if opt.data.dataset_test == 'pix3d':
        var.dpc.points[:, :, :2] *= -1
    
    if vis_only: return
    # perform per-sample alignment
    cd_acc, cd_comp, f_score = [], [], []
    for i in range(batch_size):
        best_acc, best_comp, best_fscore, best_pred, best_gt = \
            brute_force_search(var.dpc_pred[i], var.dpc.points[i], opt.eval.f_thresholds, opt.device)
        # rotate the prediction
        var.dpc_pred[i] = best_pred.clone()
        var.dpc.points[i] = best_gt.clone()
        cd_acc.append(best_acc)
        cd_comp.append(best_comp)
        f_score.append(best_fscore)
    var.cd_acc = torch.stack(cd_acc, dim=0)
    var.cd_comp = torch.stack(cd_comp, dim=0)
    var.f_score = torch.stack(f_score, dim=0)
    return var.cd_acc.mean(), var.cd_comp.mean()

def eval_metrics(opt, var, vis_only=False):
    # if opt.eval.brute_force:
    #     return eval_metrics_BF(opt, var, impl_network, vis_only)
    # else:
    #     return eval_metrics_default(opt, var, impl_network, vis_only)
    device = var.symm_outputs["cls_logits"].device
    return torch.tensor(0.).to(device), torch.tensor(0.).to(device)

    

def calculate_geodesic_distances_torch(normals1, normals2):
    # Calculate dot products between all pairs of normals
    dot_products = torch.matmul(normals1, normals2.T)
    # Clamp the values to be within [-1, 1] to avoid numerical issues with acos
    clamped_dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angles in radians
    angles_rad = torch.acos(torch.abs(clamped_dot_products))
    # Convert the angles to degrees
    geodesic_distances = torch.rad2deg(angles_rad)
    return geodesic_distances    

def eval_symm(opt, var, vis_only=False):

    conf_thres = 0.7
    # is_refined = False
    only_eval_symmetric = True
    symmf_thress = [1, 5, 10, 15, 20, 25, 30, 45, 60]
    
    outputs = var.symm_outputs
    targets = var.symm_targets
    assignments = var.symm_assignments
    device = outputs["cls_logits"].device
    batch_size = outputs["cls_logits"].shape[0]

    result = {
        "ndata":                torch.tensor(0.).to(device),
        "nsymmetric":           torch.tensor(0.).to(device),
        "npred":                torch.tensor(0.).to(device),
        "ngt":                  torch.tensor(0.).to(device),
        "num_has_pred_has_gt":  torch.tensor(0.).to(device),
        "num_has_pred_no_gt":   torch.tensor(0.).to(device),
        "num_no_pred_has_gt":   torch.tensor(0.).to(device),
        "num_no_pred_no_gt":    torch.tensor(0.).to(device),
        "center_distance":      torch.tensor(0.).to(device),
        "ngd_recall":           torch.tensor(0.).to(device),
        "ngd_precision":        torch.tensor(0.).to(device),
        "symm_tp":              torch.zeros(len(symmf_thress)).to(device),
        "symm_fp":              torch.zeros(len(symmf_thress)).to(device),
        "symm_fn":              torch.zeros(len(symmf_thress)).to(device),
        # "symm_fscores": np.zeros(len(symmf_thress))
    } # and more added at the end





    all_num_gt = targets["num_actual_gt"]
    all_probs = outputs["cls_logits"].sigmoid()
    all_normal = outputs["normal"]
    all_offset = outputs["offset"]
    all_gt_center = targets["center_coords"]
    all_gt_normal = targets["gt_normal_normalized"]

    # cls_f1
    # pred_cls: B x Q
    # pred_cls = pred_logits.argmax(-1).type(torch.int64)
    pred_logits = outputs["cls_logits"].squeeze(-1)
    pred_logits = torch.sigmoid(pred_logits)
    pred_cls = (pred_logits > THRESHOLD).type(torch.int64)
    # pred_cls = (all_probs > conf_thres).type(torch.int64)
    proposal_matched_mask = assignments["proposal_matched_mask"]
    gt_box_label = proposal_matched_mask.type(torch.int64)
    cls_tp = torch.sum((pred_cls == 1) & (gt_box_label == 1))
    cls_fp = torch.sum((pred_cls == 1) & (gt_box_label == 0))
    cls_fn = torch.sum((pred_cls == 0) & (gt_box_label == 1))

    old_result = {
        # "cardinality": cardinality,
        "tp": cls_tp,
        "fp": cls_fp,
        "fn": cls_fn
    }

    for b in range(batch_size):
        ######## need: 
        # v["num_gt"]
        # v["probs"]
        # v["normal"]
        # v["offset"]
        # v["gt_center"]
        # v["gt_normal"]
        ngt = all_num_gt[b][0]
        probs = all_probs[b]
        normal = all_normal[b]
        offset = all_offset[b]
        gt_center = all_gt_center[b]
        gt_normal = all_gt_normal[b]

        pred = probs > conf_thres
        pred = pred.flatten()
        npred = pred.sum()

        result["ndata"] += 1
        if ngt > 0:
            result["nsymmetric"] += 1
        result["ngt"] += ngt
        result["npred"] += npred

        if ngt > 0 and npred > 0:
            result["num_has_pred_has_gt"] += 1
        elif ngt > 0 and npred == 0:
            result["num_no_pred_has_gt"] += 1
            result["symm_fn"] += ngt
            continue
        elif ngt == 0 and npred > 0:
            result["num_has_pred_no_gt"] += 1
            if not only_eval_symmetric:
                result["symm_fp"] += npred
            continue
        elif ngt == 0 and npred == 0:
            result["num_no_pred_no_gt"] += 1
            continue

        pred_normals = normal[pred]
        pred_offsets = offset[pred]

        gt_center = gt_center[0]    # 3
        gt_normals = gt_normal[:ngt] # Ngt x 3        

        # 1. Distance between gt_center and each predicted plane
        distances_to_planes = torch.abs(torch.matmul(pred_normals, gt_center.T) + pred_offsets.squeeze())
        result["center_distance"] += distances_to_planes.mean()

        # 2. Geodesic distance between each predicted normal and its nearest gt normal
        cos_angles_pred_to_gt = torch.abs(torch.matmul(pred_normals, gt_normals.T))
        max_cos_angles_pred_to_gt, _ = torch.max(cos_angles_pred_to_gt, dim=1)
        geodesic_distances_pred_to_gt = torch.rad2deg(torch.acos(max_cos_angles_pred_to_gt))
        result["ngd_precision"] += geodesic_distances_pred_to_gt.sum()

        # 3. Geodesic distance between each gt normal and its nearest predicted normal
        cos_angles_gt_to_pred = torch.abs(torch.matmul(gt_normals, pred_normals.T))
        max_cos_angles_gt_to_pred, _ = torch.max(cos_angles_gt_to_pred, dim=1)
        geodesic_distances_gt_to_pred = torch.rad2deg(torch.acos(max_cos_angles_gt_to_pred))
        result["ngd_recall"] += geodesic_distances_gt_to_pred.sum()

        geodesic_distances = calculate_geodesic_distances_torch(pred_normals, gt_normals)
        for t, symmf_thres in enumerate(symmf_thress):
            tp, fp, fn = 0, 0, 0
            
            # Mark matches
            matched_gt = set()
            matched_pred = set()
            
            for i in range(geodesic_distances.shape[0]):
                for j in range(geodesic_distances.shape[1]):
                    if geodesic_distances[i, j] < symmf_thres:
                        if j not in matched_gt and i not in matched_pred:
                            tp += 1
                            matched_gt.add(j)
                            matched_pred.add(i)
            
            fp = len(pred_normals) - tp
            fn = len(gt_normals) - tp
            result["symm_tp"][t] += tp
            result["symm_fp"][t] += fp
            result["symm_fn"][t] += fn       



    ##########################################################################################
    # TODO: this assumes >1 batch size which is usually not true, move aggregation elsewhere
    ##########################################################################################

    # merge old_result and result
    for k, v in old_result.items():
        result[k] = v
    
    var.symm_eval_metric = result

    batch_aggregated_f1 = 2 * cls_tp / (2 * cls_tp + cls_fp + cls_fn + 1.e-8)
    # if result["ngt"] == 0 or result["npred"] == 0:
    #     batch_aggregated_geo = torch.tensor(0.0).to(device)
    # else:
    batch_aggregated_geo = (result["ngd_precision"] / result["npred"] + result["ngd_recall"] / result["ngt"]) / 2
    batch_aggregated_center_dist = result["center_distance"] / result["num_has_pred_has_gt"]
    batch_aggregated_symm_fat = result["symm_tp"] / (result["symm_tp"] + 0.5*(result["symm_fp"] + result["symm_fn"] + 1.e-8))
    return batch_aggregated_f1, batch_aggregated_geo, batch_aggregated_center_dist, batch_aggregated_symm_fat

def generate_symm_eval(result, opt):

    result["symm_no_symm_f1"] = result["num_has_pred_has_gt"] / (result["num_has_pred_has_gt"] + 0.5*(result["num_has_pred_no_gt"] + result["num_no_pred_has_gt"] + 1.e-8))

    result["ngd_recall"] = result["ngd_recall"] / result["ngt"] #result["num_has_pred_has_gt"]
    result["ngd_precision"] = result["ngd_precision"] / result["npred"] #result["num_has_pred_has_gt"]
    result["ngd"] = 0.5 * (result["ngd_recall"] + result["ngd_precision"])

    result["center_distance"] = result["center_distance"] / result["num_has_pred_has_gt"]

    result["symm_fscores"] = result["symm_tp"] / (result["symm_tp"] + 0.5*(result["symm_fp"] + result["symm_fn"] + 1.e-8))


    # ------------ old ----------------
    result["cls_precision"] = result["tp"] / (result["tp"] + result["fp"] + 1.e-8)
    result["cls_recall"] = result["tp"] / (result["tp"] + result["fn"] + 1.e-8)
    result["cls_f1"] = 2 * result["tp"] / (2 * result["tp"] + result["fp"] + result["fn"] + 1.e-8)

    return result

def print_symm_eval(symm_eval_final, opt):
    for k,v in symm_eval_final.items():
        print(f"{k}: {v}")


def compute_fscore(dist1, dist2, thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscores
    """
    fscores = []
    for threshold in thresholds:
        precision = torch.mean((dist1 < threshold).float(), dim=1)  # [B, ]
        recall = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore)] = 0
        fscores.append(fscore)
    fscores = torch.stack(fscores, dim=1)
    return fscores

def convert_to_explicit(opt, level_grids, isoval=0., to_pointcloud=False):
    N = len(level_grids)
    meshes = [None]*N
    pointclouds = [None]*N if to_pointcloud else None
    threads = [threading.Thread(target=convert_to_explicit_worker,
                                args=(opt, i, level_grids[i], isoval, meshes),
                                kwargs=dict(pointclouds=pointclouds),
                                daemon=False) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    if to_pointcloud:
        pointclouds = np.stack(pointclouds, axis=0)
        return meshes, pointclouds
    else: return meshes

def convert_to_explicit_worker(opt, i, level_vox_i, isoval, meshes, pointclouds=None):
    # use marching cubes to convert implicit surface to mesh
    vertices, faces = mcubes.marching_cubes(level_vox_i, isovalue=isoval)
    assert(level_vox_i.shape[0]==level_vox_i.shape[1]==level_vox_i.shape[2])
    S = level_vox_i.shape[0]
    range_min, range_max = opt.eval.range
    # marching cubes treat every cube as unit length
    vertices = vertices/S*(range_max-range_min)+range_min
    mesh = trimesh.Trimesh(vertices, faces)
    meshes[i] = mesh
    if pointclouds is not None:
        # randomly sample on mesh to get uniform dense point cloud
        if len(mesh.triangles)!=0:
            points = mesh.sample(opt.eval.num_points)
        else: points = np.zeros([opt.eval.num_points, 3])
        pointclouds[i] = points

def chamfer_distance(opt, X1, X2):
    assert(X1.shape[2]==3)
    Chamfer_3D = chamfer_3DDist().to(X1.device)
    dist_1, dist_2, idx_1, idx_2 = Chamfer_3D(X1, X2)
    return dist_1.sqrt(), dist_2.sqrt(), idx_1, idx_2

def ICP(opt, X1, X2, num_iter=50, return_cd=False): # [B, N, 3]
    assert(len(X1)==len(X2))
    for it in range(num_iter):
        d1, d2, idx, _ = chamfer_distance(opt, X1, X2)
        X2_corresp = torch.zeros_like(X1)
        for i in range(len(X1)):
            X2_corresp[i] = X2[i][idx[i].long()]
        t1 = X1.mean(dim=-2, keepdim=True)
        t2 = X2_corresp.mean(dim=-2, keepdim=True)
        U, S, V = ((X1-t1).transpose(1, 2)@(X2_corresp-t2)).svd(some=True)
        R = V@U.transpose(1, 2)
        R[R.det()<0, 2] *= -1
        X1 = (X1-t1)@R.transpose(1, 2)+t2
    if return_cd:
        d1, d2, idx, _ = chamfer_distance(opt, X1, X2)
        return X1, ((d1 + d2)/2).mean()
    else:
        return X1
