import logging
import os
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import NoopLogger

from utils import create_rllib_env

# --- Configuration ---
NUM_ENVS_PER_WORKER = 2
TRAINING_HOURS = int(os.environ.get("STRONG_TRAIN_HOURS", "24"))
TIMESTEP_TARGET = int(os.environ.get("STRONG_TRAIN_TIMESTEPS", "60000000"))
RESTORE_CHECKPOINT = os.environ.get("STRONG_RESTORE_CHECKPOINT")
BASE_PORT = int(os.environ.get("STRONG_BASE_PORT", str(15000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 40000))))

# --- Filters ---
class RayErrorFilter(logging.Filter):
    def filter(self, record):
        return not any(err in record.getMessage() for err in ["The agent on node", "socket.gaierror"])

logging.getLogger("ray._private.worker").addFilter(RayErrorFilter())
logging.getLogger("ray.worker").addFilter(RayErrorFilter())
logging.getLogger("ray").setLevel(logging.ERROR)

os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"

# --- Multi-Agent Curriculum ---
def assign_team_policy(agent_id, *args, **kwargs):
    """Maps team 1 to the learner and team 2 to historical snapshots."""
    if agent_id in (0, 1):
        return "default"
    
    # Slightly tweaked probability distribution from your classmate
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.10, 0.45, 0.30, 0.15],
    )

class HistoricalOpponentUpdate(DefaultCallbacks):
    """Snapshots the active policy into an archive for curriculum learning."""
    
    def on_train_result(self, *, trainer, result, **kwargs):
        mean_reward = result.get("episode_reward_mean", -999)
        current_iter = result.get("training_iteration", 0)

        # Initial seeding of the opponents
        if not getattr(self, "_is_seeded", False):
            active_weights = trainer.get_weights(["default"])["default"]
            trainer.set_weights({
                f"opponent_{i}": active_weights for i in range(1, 4)
            })
            self._is_seeded = True
            print("[Curriculum] Opponent pool initialized with starting weights.")

        # Snapshot promotion logic
        if current_iter > 0 and current_iter % 25 == 0 and mean_reward > -0.20:
            print(f"[Curriculum] Iteration {current_iter} (Reward: {mean_reward:.3f}) -> Promoting weights to archive.")
            trainer.set_weights({
                "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })

# --- Main Training Loop ---
if __name__ == "__main__":
    os.system("ray stop --force")
    print(f"[Run Configuration] Port: {BASE_PORT}")

    # Explicitly set GPUs to 0 for your cluster
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
            "progress_weight": 0.08,
            "territory_weight": 0.01,
            "possession_weight": 0.02,
            "defense_weight": 0.01,
            "concede_penalty": 1.0,
            "clip_abs": 0.20,
        },
        "use_ball_feature_observation": True,
        "ball_feature_observation_config": {
            "feature_clip": 1.0,
        },
    }

    # Dummy env to extract spaces
    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    stop_condition = {"timesteps_total": TIMESTEP_TARGET}
    if not RESTORE_CHECKPOINT:
        stop_condition["time_total_s"] = TRAINING_HOURS * 3600

    analysis = tune.run(
        "PPO",
        name="PPO_SelfPlay_Curriculum",
        loggers=[NoopLogger],
        restore=RESTORE_CHECKPOINT,
        config={
            "num_gpus": 0, # Hardware fix
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": HistoricalOpponentUpdate,
            "seed": 42,
            
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(assign_team_policy),
                "policies_to_train": ["default"],
            },
            
            "env": "Soccer",
            "env_config": env_config,
            
            # Tweaked Neural Network
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [512, 256, 128], # Shrunk from 768/512/256
                "fcnet_activation": "relu",       # Changed from swish
            },
            
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "vf_loss_coeff": 1.0,
            
            # Fixed Batching Math
            "rollout_fragment_length": 500,
            "train_batch_size": 14000,        # Changed from 16000 to match workers
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 16,               # Dropped from 24 to prevent overfitting
            "batch_mode": "complete_episodes",
            
            "lr": 3e-4,
            "lr_schedule": [
                [0, 3e-4],
                [15000000, 8e-5],
                [30000000, 3e-5],
            ],
        },
        stop=stop_condition,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("~/scratch/ray_results"),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(f"Training Complete. Best checkpoint: {best_checkpoint}")