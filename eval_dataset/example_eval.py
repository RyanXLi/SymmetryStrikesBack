#!/usr/bin/env python3
"""
Symmetry Prediction Evaluation Script

This script evaluates the performance of symmetry prediction models by comparing
predicted symmetry planes against ground truth data. It computes various metrics
including normal geodesic distance, precision, recall, and F-scores.

Usage:
    python example_eval.py --dataset gso --data_dir /path/to/data --symm_gt_dir /path/to/symm_gt
"""

import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def generate_symm_eval(eval_metrics: Dict) -> Dict:
    """
    Generate final symmetry evaluation metrics from aggregated results.
    
    Args:
        eval_metrics: Dictionary containing aggregated evaluation metrics
        
    Returns:
        Dictionary with normalized evaluation metrics
    """
    if eval_metrics["n_has_both"] == 0:
        # Handle case where no valid predictions exist
        return {
            "cardinality": 0,
            "normal_geodesic_distance_recall": 0,
            "normal_geodesic_distance_precision": 0,
            "normal_geodesic_distance": 0,
            "npred": eval_metrics["npred"],
            "ngt": eval_metrics["ngt"],
            "num_data": eval_metrics["num_data"],
            "n_has_both": 0,
            "num_has_pred_has_gt": eval_metrics["num_has_pred_has_gt"],
            "num_has_pred_no_gt": eval_metrics["num_has_pred_no_gt"],
            "num_no_pred_has_gt": eval_metrics["num_no_pred_has_gt"],
            "num_no_pred_no_gt": eval_metrics["num_no_pred_no_gt"],
            "symm_fscores": eval_metrics["symm_fscores"] / max(eval_metrics["num_data"], 1),
        }
    
    result = {
        "cardinality": eval_metrics["cardinality"],
        "normal_geodesic_distance_recall": eval_metrics["normal_geodesic_distance_recall"] / eval_metrics["n_has_both"],
        "normal_geodesic_distance_precision": eval_metrics["normal_geodesic_distance_precision"] / eval_metrics["n_has_both"],
        "normal_geodesic_distance": (eval_metrics["normal_geodesic_distance_recall"] + eval_metrics["normal_geodesic_distance_precision"]) / eval_metrics["n_has_both"] / 2,
        "npred": eval_metrics["npred"],
        "ngt": eval_metrics["ngt"],
        "num_data": eval_metrics["num_data"],
        "n_has_both": eval_metrics["n_has_both"],
        "num_has_pred_has_gt": eval_metrics["num_has_pred_has_gt"],
        "num_has_pred_no_gt": eval_metrics["num_has_pred_no_gt"],
        "num_no_pred_has_gt": eval_metrics["num_no_pred_has_gt"],
        "num_no_pred_no_gt": eval_metrics["num_no_pred_no_gt"],
        "symm_fscores": eval_metrics["symm_fscores"] / eval_metrics["num_data"],
    }
    return result


def print_symm_eval(symm_eval_final: Dict) -> None:
    """Print symmetry evaluation results in a formatted manner."""
    print("\n" + "="*50)
    print("SYMMETRY EVALUATION RESULTS")
    print("="*50)
    for k, v in symm_eval_final.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
            else:
                print(f"{k}: {v.cpu().numpy()}")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("="*50)


def aggregate_symm_eval(total_symm_eval: Optional[Dict], symm_eval: Dict) -> Dict:
    """
    Aggregate symmetry evaluation metrics across multiple samples.
    
    Args:
        total_symm_eval: Existing aggregated metrics (None for first sample)
        symm_eval: Current sample metrics to add
        
    Returns:
        Updated aggregated metrics
    """
    if total_symm_eval is None:
        return symm_eval.copy()
    
    for k, v in symm_eval.items():
        total_symm_eval[k] += v
    return total_symm_eval


def load_symmetric_datanames(symm_list_file: str) -> List[str]:
    """
    Load list of symmetric object names from file.
    
    Args:
        symm_list_file: Path to file containing symmetric object names
        
    Returns:
        List of symmetric object names
    """
    if not os.path.exists(symm_list_file):
        raise FileNotFoundError(f"Symmetric object list file not found: {symm_list_file}")
    
    datanames = []
    with open(symm_list_file, "r") as f:
        for line in f:
            datanames.append(line.strip())
    return datanames


def compute_geodesic_distance_metrics(symm_preds: torch.Tensor, symm_gt: torch.Tensor, 
                                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute geodesic distance metrics between predictions and ground truth.
    
    Args:
        symm_preds: Predicted symmetry normals [N, 3]
        symm_gt: Ground truth symmetry normals [M, 3]
        device: PyTorch device
        
    Returns:
        Tuple of (recall_distances, precision_distances, fscores)
    """
    # F-score thresholds in degrees
    thresholds = [0.5, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # Compute cosine similarity matrix
    normal_cos_sim = torch.mm(symm_preds, symm_gt.T)
    # Handle symmetry: replace negative values with their absolute values
    normal_cos_sim = normal_cos_sim.where(normal_cos_sim > 0, -normal_cos_sim)
    normal_cos_sim = normal_cos_sim.clamp(min=0., max=1.)
    
    # Compute geodesic distances for recall (GT coverage)
    rec_b_normal_cos_sim, _ = normal_cos_sim.max(1)  # Best match for each GT
    rec_b_normal_geodesic_dist = torch.acos(rec_b_normal_cos_sim)
    rec_b_normal_geodesic_dist = torch.rad2deg(rec_b_normal_geodesic_dist)
    
    # Compute geodesic distances for precision (prediction accuracy)
    prec_b_normal_cos_sim, _ = normal_cos_sim.max(0)  # Best match for each prediction
    prec_b_normal_geodesic_dist = torch.acos(prec_b_normal_cos_sim)
    prec_b_normal_geodesic_dist = torch.rad2deg(prec_b_normal_geodesic_dist)
    
    # Compute F-scores at different thresholds
    fscores = []
    for threshold in thresholds:
        precision = torch.mean((prec_b_normal_geodesic_dist < threshold).float())
        recall = torch.mean((rec_b_normal_geodesic_dist < threshold).float())
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = torch.tensor(0.).to(device)
        fscores.append(fscore)
    
    fscores = torch.stack(fscores, dim=0)
    
    return rec_b_normal_geodesic_dist, prec_b_normal_geodesic_dist, fscores


def create_empty_eval_metrics(ngt: int = 0, device: torch.device = None) -> Dict:
    """Create empty evaluation metrics for cases with no predictions."""
    if device is None:
        device = torch.device("cpu")
    
    return {
        "cardinality": 0,
        "normal_geodesic_distance_recall": torch.tensor(0.).to(device),
        "normal_geodesic_distance_precision": torch.tensor(0.).to(device),
        "npred": 0,
        "ngt": ngt,
        "num_data": 1,
        "n_has_both": torch.tensor(0.).to(device),
        "num_has_pred_has_gt": torch.tensor(0.).to(device),
        "num_has_pred_no_gt": torch.tensor(0.).to(device),
        "num_no_pred_has_gt": torch.tensor(1.).to(device),
        "num_no_pred_no_gt": torch.tensor(0.).to(device),
        "symm_fscores": torch.zeros(14).to(device)  # 14 thresholds
    }


def evaluate_dataset(args) -> None:
    """
    Main evaluation function for symmetry prediction dataset.
    
    Args:
        args: Command line arguments containing dataset configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load symmetric object names
    symm_list_file = os.path.join(args.symm_list_dir, f"{args.dataset}_name_list_symm.txt")
    datanames = load_symmetric_datanames(symm_list_file)
    print(f"Loaded {len(datanames)} symmetric objects for evaluation")
    
    # Get all image files from dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    sorted_names = sorted([f for f in os.listdir(args.data_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(sorted_names)} images in dataset")
    
    # Evaluate at specified confidence thresholds
    for conf_thres in args.confidence_thresholds:
        print(f"\nEvaluating with confidence threshold: {conf_thres}")
        total_symm_eval = None
        processed_count = 0
        
        for filename in tqdm(sorted_names, desc="Processing images"):
            rawname = filename.split('.')[0]
            dataname = '_'.join(rawname.split('_')[:-2])
            
            # Skip non-symmetric objects
            if dataname not in datanames:
                continue
            
            # Load ground truth symmetry
            symm_gt_file = os.path.join(args.symm_gt_dir, f"{dataname}.json")
            if not os.path.exists(symm_gt_file):
                print(f"Warning: No ground truth file for {dataname}")
                continue
            
            with open(symm_gt_file, "r") as f:
                symm_gt = json.load(f)['normals']
            
            symm_gt = torch.tensor(symm_gt).to(device)
            symm_gt = symm_gt / torch.norm(symm_gt, dim=1, keepdim=True)
            
            # Load prediction results
            pred_file = os.path.join(args.pred_dir, rawname, args.pred_subdir, "symm_out.json")
            if not os.path.exists(pred_file):
                # No prediction available
                symm_eval = create_empty_eval_metrics(len(symm_gt), device)
                total_symm_eval = aggregate_symm_eval(total_symm_eval, symm_eval)
                continue
            
            with open(pred_file, "r") as f:
                symm_out = json.load(f)
            
            # Process predictions
            symm_preds = [symm_out['normal']]
            symm_preds = np.array(symm_preds)
            symm_preds = torch.tensor(symm_preds).float().to(device)
            
            if len(symm_preds) == 0:
                symm_eval = create_empty_eval_metrics(len(symm_gt), device)
                total_symm_eval = aggregate_symm_eval(total_symm_eval, symm_eval)
                continue
            
            symm_preds = symm_preds / torch.norm(symm_preds, dim=1, keepdim=True)
            
            # Compute metrics
            num_pred = len(symm_preds)
            num_valid_gt = len(symm_gt)
            
            if num_pred == 0 or num_valid_gt == 0:
                symm_eval = create_empty_eval_metrics(num_valid_gt, device)
                total_symm_eval = aggregate_symm_eval(total_symm_eval, symm_eval)
                continue
            
            # Compute geodesic distance metrics
            rec_distances, prec_distances, fscores = compute_geodesic_distance_metrics(
                symm_preds, symm_gt, device)
            
            # Create evaluation metrics for this sample
            symm_eval = {
                "cardinality": num_pred,
                "normal_geodesic_distance_recall": rec_distances.mean(),
                "normal_geodesic_distance_precision": prec_distances.mean(),
                "npred": num_pred,
                "ngt": num_valid_gt,
                "num_data": 1,
                "n_has_both": torch.tensor(1.).to(device),
                "num_has_pred_has_gt": torch.tensor(1.).to(device),
                "num_has_pred_no_gt": torch.tensor(0.).to(device),
                "num_no_pred_has_gt": torch.tensor(0.).to(device),
                "num_no_pred_no_gt": torch.tensor(0.).to(device),
                "symm_fscores": fscores
            }
            
            total_symm_eval = aggregate_symm_eval(total_symm_eval, symm_eval)
            processed_count += 1
        
        # Generate final results
        if total_symm_eval is None:
            print("Warning: No valid samples found for evaluation")
            continue
            
        symm_eval_final = generate_symm_eval(total_symm_eval)
        print_symm_eval(symm_eval_final)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"symm_result_{args.dataset}_{conf_thres:.1f}.txt")
        
        with open(output_file, "w") as f:
            f.write(f"# Symmetry Evaluation Results for {args.dataset}\n")
            f.write(f"# Confidence Threshold: {conf_thres}\n")
            f.write(f"# Processed Samples: {processed_count}\n\n")
            for k, v in symm_eval_final.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        f.write(f"{k}: {v.item():.6f}\n")
                    else:
                        f.write(f"{k}: {v.cpu().numpy().tolist()}\n")
                else:
                    f.write(f"{k}: {v:.6f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        
        print(f"Results saved to: {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate symmetry prediction performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset name (e.g., 'gso', 'omni')"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--symm_gt_dir", 
        type=str, 
        required=True,
        help="Directory containing ground truth symmetry annotations"
    )
    
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        required=True,
        help="Directory containing prediction results"
    )
    
    parser.add_argument(
        "--symm_list_dir", 
        type=str, 
        required=True,
        help="Directory containing symmetric object name lists"
    )
    
    parser.add_argument(
        "--pred_subdir", 
        type=str, 
        default="sds_symm",
        help="Subdirectory name within prediction directory containing symm_out.json"
    )
    
    parser.add_argument(
        "--confidence_thresholds", 
        type=float, 
        nargs="+",
        default=[0.0],
        help="Confidence thresholds for evaluation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Symmetry Prediction Evaluation")
    print("="*40)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Ground truth directory: {args.symm_gt_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*40)
    
    try:
        evaluate_dataset(args)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()