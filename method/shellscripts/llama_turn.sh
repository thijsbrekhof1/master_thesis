#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --output=output_%j_llama_turn.txt
#SBATCH --error=error_%j_llama_turn.txt
#SBATCH --job-name=llama_turn

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=true

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
module load GCC/12.3.0

export HUGGING_FACE_TOKEN=

source $HOME/Thijs/thijs_venv/bin/activate
pip install -r requirements.txt

python test_questions.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_source interact \
    --batch_size 4096 \
    --checkpoint_freq 4096 \
    --output_dir "results/interact/llama_turn_only" \
    --output_length 270

deactivate