# Symmetry Prediction Evaluation

This repository contains tools for evaluating symmetry prediction models by comparing predicted symmetry planes against ground truth annotations.


## File Structure

The script expects the following file structure:

```
datasets/
├── data/
│   └── gso/
│       ├── object1_view1_angle1.png
│       ├── object1_view2_angle2.jpg
│       └── ...
├── symm_gt/
│   └── gso/
│       ├── object1.json
│       ├── object2.json
│       └── ...
└── ...
```

### Ground Truth Format

Ground truth files should be JSON files with the following structure:

```json
{
    "normals": [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
}
```

### Prediction Format

Prediction files (`symm_out.json`) should contain:

```json
{
    "normal": [0.1, 0.9, 0.1],
    "confidence": 0.85
}
```

### Symmetric Object List Format

The symmetric object list file should contain one object name per line:

```
object1
object2
object3
```



## Script

The evaluation script (`example_eval.py`) computes various metrics to assess the quality of symmetry predictions:

- **Normal Geodesic Distance**: Measures angular distance between predicted and ground truth symmetry plane normals
- **Precision & Recall**: Coverage metrics for symmetry predictions
- **F-scores**: Harmonic mean of precision and recall at multiple angular thresholds
- **Cardinality**: Number of predicted symmetry planes

Note: In this example we show evaluation against an empty prediction.

## Requirements

```bash
pip install torch numpy tqdm
```

## Usage

### Basic Usage

```bash
python example_eval.py \
    --dataset gso \
    --data_dir /path/to/images \
    --symm_gt_dir /path/to/ground_truth \
    --pred_dir /path/to/predictions \
    --symm_list_dir /path/to/symmetric_object_lists
```

### Arguments

- `--dataset`: Dataset name (e.g., 'gso', 'abc')
- `--data_dir`: Directory containing input images (.png, .jpg, .jpeg)
- `--symm_gt_dir`: Directory containing ground truth symmetry annotations (JSON files)
- `--pred_dir`: Directory containing prediction results
- `--symm_list_dir`: Directory containing symmetric object name lists
- `--pred_subdir`: Subdirectory within prediction directory containing `symm_out.json` (default: 'sds_symm')
- `--confidence_thresholds`: List of confidence thresholds for evaluation (default: [0.0])
- `--output_dir`: Directory to save evaluation results (default: './evaluation_results')

### Example

Evaluation against empty prediction:

```bash
python example_eval.py \
    --dataset gso \
    --data_dir ./data/gso \
    --symm_gt_dir ./symm_gt/gso \
    --symm_list_dir ./namelist \
    --confidence_thresholds 0.0 \
    --output_dir ./results
```
