#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --output=output_clustering.txt
#SBATCH --error=error_clustering.txt
#SBATCH --job-name=clustering
#SBATCH --partition=gpu_a100

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
module load GCC/12.3.0

source $HOME/Thijs/thijs_venv/bin/activate
pip install -r requirements.txt

python cluster_discussions.py

deactivate