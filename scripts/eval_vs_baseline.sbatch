#!/bin/bash
#SBATCH --job-name=eval_mappo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_mappo-%j.out

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    module load anaconda3
fi
conda activate soccertwos

cd /home/hice1/nparikh44/cs-8803-drl/deep-football

# Ensure the logs directory exists
mkdir -p logs

# Export the path to your best MAPPO checkpoint
# (Update this path to match your actual directory structure)
export MAPPO_CHECKPOINT="./ray_results/AgentC_MAPPO_LowPBRS/PPO_Soccer_25540_00000_0_2026-04-23_10-38-51/checkpoint_001250/checkpoint-1250"

# Execute the evaluation script pitting your agent against the random baseline
python evaluate_vs_baseline.py \
  --agent1 mappo_pbrs_agent \
  --agent2 ceia_baseline_agent \
  --episodes 100