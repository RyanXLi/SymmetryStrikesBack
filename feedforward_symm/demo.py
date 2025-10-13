'''
python demo.py --yaml=options/shape.yaml --task=shape --datadir=examples --eval.vox_res=128 --ckpt= 
'''

import torch
import torchvision.transforms.functional as torchvision_F
import numpy as np
import os
import sys
import shutil
import importlib
import cv2
import utils.options as options
import utils.util_vis as util_vis

from utils.util import EasyDict as edict
from PIL import Image
from utils.eval_3D import get_dense_3D_grid, compute_level_grid, convert_to_explicit
from tqdm import tqdm
import json

from utils.eval_3D import reflect_pc, ICP, flip_mesh
import trimesh
import pyvista as pv

def get_batch_symm_dict(opt, var):
    with torch.no_grad():
        symm_dict = {}
        batch_size = var.idx.shape[0]
        offsets = var.symm_outputs['offset']
        normals = var.symm_outputs['normal']
        probs = var.symm_outputs['cls_logits'].sigmoid()
        for b in range(batch_size):
            key = var.idx[b].item()
            symm_dict[key] = {}
            symm_dict[key]['offset'] = offsets[b].detach().cpu()#.numpy()#.tolist()
            symm_dict[key]['normal'] = normals[b].detach().cpu()#.numpy()#.tolist()
            symm_dict[key]['probs'] = probs[b].detach().cpu()#.numpy()#.tolist()

        return symm_dict
    

def get_1d_bounds(arr):
        nz = np.flatnonzero(arr)
        return nz[0], nz[-1]

def get_bbox_from_mask(mask, thr):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1, y1

def square_crop(image, bbox, crop_ratio=1.):
    x1, y1, x2, y2 = bbox
    h, w = y2-y1, x2-x1
    yc, xc = (y1+y2)/2, (x1+x2)/2
    S = max(h, w)*1.2
    scale = S*crop_ratio
    image = torchvision_F.crop(image, top=int(yc-scale/2), left=int(xc-scale/2), height=int(scale), width=int(scale))
    return image

def preprocess_image(opt, image, bbox):
    image = square_crop(image, bbox=bbox)
    if image.size[0] != opt.W or image.size[1] != opt.H:
        image = image.resize((opt.W, opt.H))
    image = torchvision_F.to_tensor(image)
    rgb, mask = image[:3], image[3:]
    if opt.data.bgcolor is not None:
        # replace background color using mask
        rgb = rgb * mask + opt.data.bgcolor * (1 - mask)
        mask = (mask > 0.5).float()
    rgb = torchvision_F.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # added with dino
    return rgb, mask

# Function to apply threshold on alpha channel and create mask
def create_alpha_mask(image, alpha_threshold=128):
    alpha = image.split()[-1]  # Get the alpha channel
    alpha = np.array(alpha)
    mask = (alpha > alpha_threshold) * 255  # Apply threshold
    mask_image = Image.fromarray(mask.astype(np.uint8))
    return mask_image

def get_image(opt, image_name):
    image_fname = image_name #os.path.join(opt.datadir, 'images', image_name)

    # Define the size to which images will be resized
    new_size = (224, 224)

    image = Image.open(image_fname).convert("RGB")

    # Resize the image
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create and save the alpha mask
    alpha_mask = create_alpha_mask(image)
    mask_np = np.array(alpha_mask)
    
    image = Image.merge("RGBA", (*image.split(), alpha_mask))
    bbox = get_bbox_from_mask(mask_np, 0.5)
    rgb_input_map, mask_input_map = preprocess_image(opt, image, bbox=bbox)
    return rgb_input_map, mask_input_map

def erode_mask(mask, iterations=5):
    # [H, W]
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    mask_eroded = cv2.erode(mask_np, np.ones((3,3), np.uint8), iterations=iterations)
    mask_eroded = torch.tensor(mask_eroded).unsqueeze(0).float()
    return mask_eroded

def get_intr(opt):
    # load camera
    f = 1.3875
    K = torch.tensor([[f*opt.W, 0, opt.W/2],
                      [0, f*opt.H, opt.H/2],
                      [0, 0, 1]]).float()
    return K

def get_pixel_grid(H, W, device='cuda'):
    y_range = torch.arange(H, dtype=torch.float32).to(device)
    x_range = torch.arange(W, dtype=torch.float32).to(device)
    Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
    Z = torch.ones_like(Y).to(device)
    xyz_grid = torch.stack([X, Y, Z],dim=-1).view(-1,3) 
    return xyz_grid

def unproj_depth(depth, intr):
    '''
    depth: [B, H, W]
    intr: [B, 3, 3]
    '''
    batch_size, H, W = depth.shape
    intr = intr.to(depth.device)
    
    # [B, 3, 3]
    K_inv = torch.linalg.inv(intr).float()
    # [1, H*W,3]
    pixel_grid = get_pixel_grid(H, W, depth.device).unsqueeze(0)
    # [B, H*W,3]
    pixel_grid = pixel_grid.repeat(batch_size, 1, 1)
    # [B, 3, H*W]
    ray_dirs = K_inv @ pixel_grid.permute(0, 2, 1).contiguous()
    # [B, H*W, 3], in camera coordinates
    seen_points = ray_dirs.permute(0, 2, 1).contiguous() * depth.view(batch_size, H*W, 1)
    # [B, H, W, 3]
    seen_points = seen_points.view(batch_size, H, W, 3)
    return seen_points

def prepare_data(opt):
    # datadir = opt.datadir
    path = opt.input_image_path

    rgb_input_map, mask_input_map = get_image(opt, path)
    intr = get_intr(opt)
    # prepare data
    var = edict()
    var.rgb_input_map = rgb_input_map.unsqueeze(0).to(opt.device)
    var.mask_input_map = mask_input_map.unsqueeze(0).to(opt.device)
    var.intr = intr.unsqueeze(0).to(opt.device)
    var.idx = torch.tensor([1]).to(opt.device).long()
    var.pose_gt = False

    return [var], [path.split('/')[-1][:-4]]

def main():
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd, safe_check=False)
    opt.device = 0
    
    print("input:", opt.input_image_path)
    print("output:", opt.output_dir)

    os.makedirs(opt.output_dir, exist_ok=True)

    # build model
    task_ckpt = opt.yaml.split('/')[-1].split('.')[0]
    if task_ckpt != opt.task:
        raise ValueError('Detected different tasks between specified and the yaml, please double check!')
    if opt.task == 'shape':
        opt.pretrain.depth = None
    opt.arch.depth.pretrained = None
    module = importlib.import_module("model.compute_graph.graph_{}".format(opt.task))
    graph = module.Graph(opt).to(opt.device)
    
    # load checkpoint
    checkpoint = torch.load(opt.ckpt, map_location=torch.device(opt.device))
    ep, it, best_val = checkpoint["epoch"], checkpoint["iter"], checkpoint["best_val"]
    print("resuming from epoch {} (iteration {}, best_val {:.4f})".format(ep+1, it, best_val))
    graph.load_state_dict(checkpoint["graph"], strict=True)
    graph.eval()
    print('==> checkpoint loaded')
    
    # load the data
    data_list, name_list = prepare_data(opt)
    # print('==> sample data loaded from folder: {}'.format(opt.datadir))
    
    # create the save dir
    save_folder = opt.output_dir
    opt.output_path = opt.output_dir
    
    progress_bar = tqdm(data_list)
    for i, var in enumerate(progress_bar):
        # forward
        with torch.no_grad():
            var = graph.forward(opt, var, training=False, get_loss=False)
            if opt.task == 'shape':
                batch_symm_dict = get_batch_symm_dict(opt, var)

                # save the result
                save_refined_dict_file = os.path.join(f"{opt.output_path}", f'symm.json')
                save_symm_dict = {}
                for k,v in batch_symm_dict.items():
                    save_symm_dict[k] = {}
                    for kk, vv in v.items():
                        save_symm_dict[k][kk] = vv.numpy().tolist()
                with open(save_refined_dict_file, 'w') as f:
                    json.dump(save_symm_dict, f)    

        
if __name__ == "__main__":
    main()
