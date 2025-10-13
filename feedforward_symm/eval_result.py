import json
import sys
from tqdm import tqdm
import numpy as np

# python eval_result.py output_save/save-refined/synthetic_refined_results.json

# Read synthetic_refined_result.json
result_file = sys.argv[1]
with open(result_file, 'r') as file:
    json_data = json.load(file)

conf_thress = [0.1 * x for x in range(1, 10)]
cd_thress = [None]
is_refineds = [False, True]

symmf_thress = [1, 5, 10, 15, 20, 25, 30, 45, 60]
# dedup_thres = cosine 0.5 degree
dedup_thres = np.cos(np.deg2rad(0.5))

only_eval_symmetric = True

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

configs = [(conf_thres, cd_thres, is_refined) for is_refined in is_refineds for conf_thres in conf_thress for cd_thres in cd_thress]
print("configs:")
print(configs)

final_result = {}

# Calculate geodesic distances in degrees
def calculate_geodesic_distances(normals1, normals2):
    geodesic_distances = np.zeros((normals1.shape[0], normals2.shape[0]))
    for i, normal1 in enumerate(normals1):
        for j, normal2 in enumerate(normals2):
            cos_angle = np.abs(np.dot(normal1, normal2))
            geodesic_distances[i, j] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return geodesic_distances

for conf_thres, cd_thres, is_refined in tqdm(configs):
    print(f"computing config: conf_thres={conf_thres}, cd_thres={cd_thres}, is_refined={is_refined}...")
    result = {
        "ndata": 0,
        "nsymmetric": 0,
        "npred": 0,
        "ngt": 0,
        "num_has_pred_has_gt": 0,
        "num_has_pred_no_gt": 0,
        "num_no_pred_has_gt": 0,
        "num_no_pred_no_gt": 0,
        "center_distance": 0,
        "ngd_recall": 0,
        "ngd_precision": 0,
        "symm_tp": np.zeros(len(symmf_thress)),
        "symm_fp": np.zeros(len(symmf_thress)),
        "symm_fn": np.zeros(len(symmf_thress)),
        # "symm_fscores": np.zeros(len(symmf_thress))
    } # and more added at the end

    for k, v in tqdm(json_data.items()):

        ngt = v["num_gt"]

        if not is_refined:
            probs = np.array(v["probs"])
        else:
            probs = np.array(v["refined_conf"])
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

        # from here on, ngt > 0 and npred > 0
        pred = pred.flatten()
        if not is_refined:
            pred_normals = np.array(v["normal"])[pred]
            pred_offsets = np.array(v["offset"])[pred]
        else:
            pred_normals = np.array(v["refined_normal"])[pred]
            pred_offsets = np.array(v["refined_offset"])[pred]

        gt_center = np.array(v["gt_center"][0])    # 3
        gt_normals = np.array(v["gt_normal"][:ngt]) # Ngt x 3

        # 1. Distance between gt_center and each predicted plane
        distances_to_planes = np.abs(pred_normals @ gt_center.T + pred_offsets.squeeze())
        result["center_distance"] += distances_to_planes.mean()

        # 2. Geodesic distance between each predicted normal and its nearest gt normal
        geodesic_distances_pred_to_gt = np.zeros(pred_normals.shape[0])
        for i, pred_normal in enumerate(pred_normals):
            cos_angles = np.abs(np.dot(gt_normals, pred_normal))
            geodesic_distances_pred_to_gt[i] = np.degrees(np.arccos(np.max(cos_angles)))
        result["ngd_precision"] += geodesic_distances_pred_to_gt.sum()#.mean()

        # 3. Geodesic distance between each gt normal and its nearest predicted normal
        geodesic_distances_gt_to_pred = np.zeros(gt_normals.shape[0])
        for i, gt_normal in enumerate(gt_normals):
            cos_angles = np.abs(np.dot(pred_normals, gt_normal))
            geodesic_distances_gt_to_pred[i] = np.degrees(np.arccos(np.max(cos_angles)))
        result["ngd_recall"] += geodesic_distances_gt_to_pred.sum()

        geodesic_distances = calculate_geodesic_distances(pred_normals, gt_normals)
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
    
    
    result["symm_no_symm_f1"] = result["num_has_pred_has_gt"] / (result["num_has_pred_has_gt"] + 0.5*(result["num_has_pred_no_gt"] + result["num_no_pred_has_gt"] + 1.e-8))

    result["ngd_recall"] = result["ngd_recall"] / result["ngt"] #result["num_has_pred_has_gt"]
    result["ngd_precision"] = result["ngd_precision"] / result["npred"] #result["num_has_pred_has_gt"]
    result["ngd"] = 0.5 * (result["ngd_recall"] + result["ngd_precision"])

    result["center_distance"] = result["center_distance"] / result["num_has_pred_has_gt"]

    result["symm_fscores"] = result["symm_tp"] / (result["symm_tp"] + 0.5*(result["symm_fp"] + result["symm_fn"] + 1.e-8))
    result["symm_tp"] = result["symm_tp"].tolist()
    result["symm_fp"] = result["symm_fp"].tolist()
    result["symm_fn"] = result["symm_fn"].tolist()
    result["symm_fscores"] = result["symm_fscores"].tolist()
    for k in result:
        if type(result[k]) == np.int64:
            result[k] = int(result[k])
    final_result[f"{conf_thres}_{cd_thres}_{is_refined}"] = result

print(final_result)


# Write final result to a file
output_file = "./final_symm_result.json"
with open(output_file, 'w') as file:
    json.dump(final_result, file, indent=4)
print("Final result saved to:", output_file)