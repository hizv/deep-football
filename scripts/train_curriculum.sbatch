#!/bin/bash
#SBATCH --job-name=soccer_curriculum
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --output=logs/soccer_curr-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nparikh44@gatech.edu

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    module load anaconda3
fi
conda activate soccertwos

cd /home/hice1/nparikh44/cs-8803-drl/deep-football

mkdir -p logs

# Clean up zombie processes to prevent "Address already in use"
pkill -u $USER -f "SoccerTwos"
pkill -u $USER -f "unity"
ray stop --force

# Path to your YAML file
export CURRICULUM_YAML_PATH="./curriculum.yaml"

# We start fresh for curriculum to avoid "own-goal" bias from the 3.5M agent
python curriculum_agent/train.py