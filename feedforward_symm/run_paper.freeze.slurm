#!/bin/bash
#SBATCH --mem=188g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8
#SBATCH --time=2-00:00:00
#SBATCH --account=bckh-delta-gpu
#SBATCH --job-name=paper.freeze
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --no-requeue
###SBATCH --gpu-bind=none     # <- or closest

conda activate your_env

cd /path/to/your/repo
srun python train.py --yaml=options/shape.onlysymm.4a100.freeze.yaml --name=paper.freeze
