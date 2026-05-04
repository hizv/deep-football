import logging
import os
import sys
from pathlib import Path

# Ensure imports from repository root work regardless of launch directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ray
from ray import tune
from ray.tune.logger import NoopLogger

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 2
TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "12"))
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "25000000")) # 25M Budget

RESTORE_CHECKPOINT = None # Start entirely fresh

DEFAULT_BASE_PORT = 15000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 40000)
BASE_PORT = int(os.environ.get("STRONG_BASE_PORT", str(DEFAULT_BASE_PORT)))

class HideAgentCrashFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "The agent on node" in msg or "socket.gaierror" in msg:
            return False
        return True

logging.getLogger("ray._private.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray.worker").addFilter(HideAgentCrashFilter())
logging.getLogger("ray").setLevel(logging.ERROR)

os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"

def policy_mapping_fn(agent_id, *args, **kwargs):
    # Pure Symmetric Self-Play: The fastest way to learn defense is to fight yourself.
    return "default"

if __name__ == "__main__":
    os.system("ray stop --force")
    print(f"[Config] STRONG_BASE_PORT={BASE_PORT}")

    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=16,
        num_gpus=0,
    )

    tune.registry.register_env("Soccer", create_rllib_env)
    
    # --- 1. THE ANTI-FARMING REWARD SHAPING ---
    env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "base_port": BASE_PORT,
        "use_ball_progress_reward": True,
        "ball_progress_reward_config": {
            "progress_weight": 0.10,     # Reduced slightly to prevent exploitation
            "territory_weight": 0.05,    # High reward for keeping ball in enemy half
            "possession_weight": 0.01,   # Gutted. Touching the ball isn't enough anymore.
            "defense_weight": 0.05,      
            "concede_penalty": 1.0,      
            "clip_abs": 0.10,            # CRITICAL: Hard cap on shaping. Must score to get +1.0.
        },
        "use_ball_feature_observation": True,
        "ball_feature_observation_config": {
            "feature_clip": 1.0,
        },
    }

    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    stop_config = {
        "timesteps_total": TIMESTEP_TARGET,
        "time_total_s": TRAINING_HOURS * 3600
    }

    print(f"Starting Efficient Budget Self-Play for {TRAINING_HOURS} hours / {TIMESTEP_TARGET} steps.")

    analysis = tune.run(
        "PPO",
        name="PPO_CEIA_SpeedRun",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config={
            "num_gpus": 0,
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "seed": 42,
            
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"], 
            },
            "env": "Soccer",
            "env_config": env_config,
            
            # --- 2. THE SLIMMED NETWORK ---
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [512, 256], # Faster backward passes, quicker convergence
                "fcnet_activation": "swish",
                "use_attention": False,
                "framestack": True,          
                "num_framestacks": 3,
            },
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2, 
            "entropy_coeff": 0.01, 
            "vf_loss_coeff": 1.0,
            
            # --- 3. AGGRESSIVE SGD MATH ---
            "rollout_fragment_length": 500,
            "train_batch_size": 14000,   # 14 workers * 2 envs * 500
            "sgd_minibatch_size": 1400,  # Exactly 10 chunks per batch
            "num_sgd_iter": 15,          # Squeeze more learning out of every batch to save timesteps
            "batch_mode": "complete_episodes",
            
            # --- 4. THE "HOT START" LR SCHEDULE ---
            "lr": 3e-4,
            "lr_schedule": [
                [0, 3e-4],          # Start hot and aggressive
                [10000000, 1e-4],   # Settle down at 10M
                [20000000, 5e-5],   # Fine-tune the baseline-beating strategy
            ],
        },
        stop=stop_config,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("./ray_results"),
    )