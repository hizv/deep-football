import logging
import os
import ray
from ray import tune
from ray.tune.logger import NoopLogger

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 2
TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "12")) 
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "40000000"))
RESTORE_CHECKPOINT = None # START FRESH. Purge the own-goal farmer.

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
    # THE WINNING FIX: Role Separation
    # Agent 0 and 2 are usually spawned closer to the midfield. They become Strikers.
    # Agent 1 and 3 are usually spawned closer to the goal. They become Goalies.
    str_id = str(agent_id)
    if str_id in ("0", "2", "player_0", "player_2"):
        return "striker_policy"
    else:
        return "goalie_policy"

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
    
    env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "base_port": BASE_PORT,
        "use_ball_progress_reward": True,
        "ball_progress_reward_config": {
            "progress_weight": 0.15,
            "territory_weight": 0.01,
            "possession_weight": 0.05,
            "defense_weight": 0.05,
            "concede_penalty": 1.0,
            "clip_abs": 0.30,
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

    print(f"Starting fresh Striker/Goalie training for {TRAINING_HOURS} hours.")

    analysis = tune.run(
        "PPO",
        name="PPO_Beat_Baseline",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config={
            "num_gpus": 0,
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "seed": 42,
            
            # --- The Role Separation Setup ---
            "multiagent": {
                "policies": {
                    "striker_policy": (None, obs_space, act_space, {}),
                    "goalie_policy": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                # We train both simultaneously. They will learn to cooperate out of necessity.
                "policies_to_train": ["striker_policy", "goalie_policy"], 
            },
            "env": "Soccer",
            "env_config": env_config,
            
            # --- Network & Hyperparameters ---
            "model": {
                "vf_share_layers": False,
                # Increased slightly to handle the complexity of specialized roles
                "fcnet_hiddens": [512, 512, 256], 
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
            
            # --- Math Fixes ---
            # 14 workers * 2 envs * 500 length = 14,000 batch size.
            # 14000 is not perfectly divisible by 1024 (13.67). Changed to 1000 so it divides evenly (14).
            "rollout_fragment_length": 500,
            "train_batch_size": 14000,  
            "sgd_minibatch_size": 1000, 
            "num_sgd_iter": 10, 
            "batch_mode": "complete_episodes",
            
            "lr": 1e-4,
            "lr_schedule": [
                [0, 1e-4],
                [15000000, 6e-5],   
                [30000000, 1e-5],   
            ],
        },
        stop=stop_config,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("./ray_results"),
    )