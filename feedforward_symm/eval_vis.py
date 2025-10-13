import json
import sys
from tqdm import tqdm
import numpy as np
import pyvista as pv
import torch

# python eval_vis.py 

idx = "2"
conf_thres = 0.5
dedup_thres = np.cos(np.deg2rad(0.5))

# Read data_list.txt
with open('output_save/save-refined/data_list.txt', 'r') as file:
    data_list = file.read().splitlines()

data_dict = {}
for line in data_list:
    k, image_path, mask_path, pc_path = line.split(' data')
    data_dict[k] = {
        "image_path": "data" + image_path,
        "mask_path": "data" + mask_path,
        "pc_path": "data" + pc_path
    }

print(data_dict[idx])



# Read synthetic_refined_result.json
result_file = "output_save/save-refined/synthetic_refined_results.json"
with open(result_file, 'r') as file:
    json_data = json.load(file)

print("de-duplication of refined results...")
for k in tqdm(json_data):
    if len(json_data[k]["refined_normal"]) == 0:
        continue
    normals = np.array(json_data[k]["refined_normal"])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals.astype(np.float32)
    cos_sim = np.dot(normals, normals.T)
    cos_sim = np.triu(cos_sim, k=1)
    cos_sim = np.abs(cos_sim)
    dup_mask = cos_sim > dedup_thres
    dup_mask = np.any(dup_mask, axis=1)
    
    json_data[k]["refined_normal"] = np.array(json_data[k]["refined_normal"])[~dup_mask].tolist()
    json_data[k]["refined_conf"] = np.array(json_data[k]["refined_conf"])[~dup_mask].tolist()
    json_data[k]["refined_cd"] = np.array(json_data[k]["refined_cd"])[~dup_mask].tolist()
    json_data[k]["refined_offset"] = np.array(json_data[k]["refined_offset"])[~dup_mask].tolist()


print(json_data[idx])

# Read point cloud
pc_path = data_dict[idx]["pc_path"]
pc = np.load(pc_path)
pc = torch.from_numpy(pc).contiguous().cuda()
eg_intr_path = data_dict[idx]["mask_path"].replace("masks", "camera_data/intr").replace("png", "npy")
eg_extr_path = data_dict[idx]["mask_path"].replace("masks", "camera_data/extr").replace("png", "npy")
intr = torch.from_numpy(np.load(eg_intr_path, allow_pickle=True)).unsqueeze(0).cuda()
extr = torch.from_numpy(np.load(eg_extr_path, allow_pickle=True)).unsqueeze(0).cuda()
extr = extr.transpose(-1, -2)
pc = torch.cat([pc, torch.ones_like(pc[..., :1])], dim=-1)
pc = torch.matmul(pc, extr)
pc -= pc.mean(dim=1, keepdim=True)

pc = pc.squeeze().cpu().numpy()



gt_center = json_data[idx]['gt_center']
gt_normals = json_data[idx]['gt_normal']


pred_refined = np.array(json_data[idx]['refined_conf']) > conf_thres
pred_refined = pred_refined.flatten()
refined_normals = np.array(json_data[idx]['refined_normal'])[pred_refined]
refined_normals = refined_normals / np.linalg.norm(refined_normals, axis=1, keepdims=True)
refined_normals = refined_normals.tolist()


pred_raw = np.array(json_data[idx]['probs']) > conf_thres
pred_raw = pred_raw.flatten()
raw_normals = np.array(json_data[idx]['normal'])[pred_raw]
raw_normals = raw_normals / np.linalg.norm(raw_normals, axis=1, keepdims=True)
raw_normals = raw_normals.tolist()

plotter = pv.Plotter()
plotter.show_axes()
plotter.view_xy()

# plot origin in yellow
plotter.add_mesh(pv.Sphere(radius=0.1), color='yellow')

pvpc_orig = pv.PolyData(pc)
plotter.add_mesh(pvpc_orig, color='green')

for gt_normal in gt_normals:
    plotter.add_mesh(pv.Plane(center=[0,0,0], direction=gt_normal, i_size=10, j_size=10), color='green')
for n in refined_normals:
    plotter.add_mesh(pv.Plane(center=[0,0,0], direction=n, i_size=10, j_size=10), color='blue')
for n in raw_normals:
    plotter.add_mesh(pv.Plane(center=[0,0,0], direction=n, i_size=10, j_size=10), color='red')
plotter.show()
