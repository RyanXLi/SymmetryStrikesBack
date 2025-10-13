# Symmetry-Aware 3D Reconstruction

Fork of [ZeroShape](https://github.com/zxhuang1698/ZeroShape) with feed-forward symmetry detection.

## Setup

```bash
conda env create -f requirements.yaml
```

Install additional requirements:
```bash
pip install -r additional_req.txt
```

## Data & Checkpoints

- **Checkpoint**: Download from [Google Drive](https://drive.google.com/file/d/1rXTDZzV4-TrB36X1Ny9yu7ljCcV-ntWn/view?usp=sharing) and place in `weights/`
- **Training data**: Derived from ZeroShape's training data. Download from [Dropbox](https://www.dropbox.com/scl/fi/ac775o7n6wzzwvbw66cae/combined_synthetic.tar.gz?rlkey=c635jixlul9of8un2aw43xpq7&dl=0) following [ZeroShape's instructions](https://github.com/zxhuang1698/ZeroShape). Symmetry ground truth included in `data/symm/new_symm_gt.json`.

## Usage

**Inference**: `bash demo.sh`

**Training**: `sbatch run_paper.freeze.slurm` or `python train.py --config options/shape.onlysymm.4a100.freeze.yaml`

