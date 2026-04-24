import argparse
import logging
import os
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import NoopLogger

from utils import create_rllib_env
from soccer_twos import EnvType

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
    if agent_id in (0, 1):
        return "default"

    # Team 2 plays from a weighted opponent pool.
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.20, 0.40, 0.25, 0.15],
    )

class SelfPlayArchiveCallback(DefaultCallbacks):
    """
    Periodically promotes the latest trainable policy into opponent archive.
    """
    def on_train_result(self, *, trainer, result, **kwargs):
        reward_mean = result.get("episode_reward_mean", -999)
        iteration = result.get("training_iteration", 0)

        if not getattr(self, "_archive_seeded", False):
            default_weights = trainer.get_weights(["default"])["default"]
            trainer.set_weights(
                {
                    "opponent_1": default_weights,
                    "opponent_2": default_weights,
                    "opponent_3": default_weights,
                }
            )
            self._archive_seeded = True
            print("[SelfPlayArchive] seeded opponent archive from default policy")

        # Rotate archive more frequently once the policy reaches stable play
        if iteration > 0 and iteration % 25 == 0 and reward_mean > -0.20:
            print(
                f"[SelfPlayArchive] iter={iteration}, reward={reward_mean:.3f} -> rotating opponent weights"
            )
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=60_000_000)
    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=15000)
    parser.add_argument("--experiment-name", type=str, default="PPO_Vanilla_HugeNet")
    args = parser.parse_args()

    # Dynamic base port for slurm compatibility
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is not None and slurm_job_id.isdigit():
        base_port = 15000 + (int(slurm_job_id) % 40000)
    else:
        base_port = args.base_port

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
    )

    tune.registry.register_env("Soccer", create_rllib_env)
    
    env_config = {
        "variation": EnvType.multiagent_player,
        "base_port": base_port,
        "num_envs_per_worker": args.num_envs_per_worker,
        "use_compact_obs": True, 
        "return_dict_obs": False,
        "use_pbrs": True,        
        "pbrs_scale": 1.0,
        "pbrs_gamma": 0.99,
    }
    
    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": SelfPlayArchiveCallback,
            "seed": 42,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": env_config,
            
            # Massive Neural Network parameters
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [768, 512, 256],
                "fcnet_activation": "swish",
            },
            
            # RLlib algorithmic hyperparameters
            "lambda": 0.95,
            "gamma": 0.99,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "vf_loss_coeff": 1.0,
            
            # Massive Batching setup
            "rollout_fragment_length": 500,
            "train_batch_size": 16000,
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 24,
            "batch_mode": "complete_episodes",
            
            # Adaptive learning rate schedule
            "lr": 3e-4,
            "lr_schedule": [
                [0, 3e-4],
                [15_000_000, 8e-5],
                [30_000_000, 3e-5],
            ],
        },
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=10,
        keep_checkpoints_num=5,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        local_dir=os.path.expanduser("./ray_results"),
    )
    
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max", scope="all")
    if best_trial is not None:
        print(f"Best trial: {best_trial}")
        best_checkpoint = analysis.get_best_checkpoint(
            best_trial, "episode_reward_mean", mode="max"
        )
        print(f"Best checkpoint: {best_checkpoint}")
    else:
        print("No best trial found.")

if __name__ == "__main__":
    main()