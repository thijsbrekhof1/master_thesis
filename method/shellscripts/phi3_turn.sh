#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --output=output_%j_phi3_turn.txt
#SBATCH --error=error_%j_phi3_turn.txt
#SBATCH --job-name=phi3_turn
#SBATCH --partition=gpu_a100


export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=true

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
module load GCC/12.3.0

source $HOME/Thijs/thijs_venv/bin/activate
pip install -r requirements.txt

python test_questions.py \
    --model microsoft/Phi-3-medium-128k-instruct \
    --data_source interact \
    --batch_size 2048 \
    --checkpoint_freq 2048 \
    --output_dir "results/interact/phi3_turn_only" \
    --output_length 270

deactivate