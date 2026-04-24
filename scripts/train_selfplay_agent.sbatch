#!/bin/bash
#SBATCH --job-name=soccer_speedrun_Selfplay
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=2G
#SBATCH -t 12:00:00
#SBATCH --output=logs/soccer_speedrun_Selfplay-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nparikh44@gatech.edu

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    module load anaconda3
fi
conda activate soccertwos

cd /home/hice1/nparikh44/cs-8803-drl/deep-football

# Make sure the logs directory exists so the job doesn't fail on startup
mkdir -p logs

# Environment variables read by the Python script
export STRONG_TRAIN_HOURS=12
export STRONG_TRAIN_TIMESTEPS=40000000

# Execute the training script
python selfplay_speedrun/train.py