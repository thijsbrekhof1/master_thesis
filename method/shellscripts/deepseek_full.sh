#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=deepseek_processing
#SBATCH --output=output_%j_deepseek_full.txt
#SBATCH --error=error_%j_deepseek_full.txt

# Load required modules
module purge
module load 2024
module load Python/3.11.3-GCCcore-12.3.0
module load GCC/12.3.0

source $HOME/Thijs/thijs_venv/bin/activate
pip install -r requirements.txt

# Set API key as environment variable
export DEEPSEEK_API_KEY=

# Run the script
python test_questions_deepseek.py \
    --data_source rfc \
    --checkpoint_freq 500 \
    --use_full_context



deactivate
