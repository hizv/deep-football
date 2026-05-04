"""Train PPO directly against ceia_baseline_agent as a frozen opponent.

Same-day path to beat the baseline in eval:
  - Matches the baseline's architecture ([256, 256] with vf_share_layers).
  - Learner policy plays as team 0 (agent ids 0, 1); frozen baseline plays
    team 1 (agent ids 2, 3). The baseline policy is loaded from its Ray
    checkpoint and its weights are held fixed (policies_to_train=["default"]).
  - Self-play-style opponent diversity is added via three lagged snapshots of
    the learner, mixed with the live baseline. This hedges against
    overfitting to the baseline's specific weaknesses.

Usage:
  python train_ppo_vs_baseline.py

Then evaluate with:
  python -m soccer_twos.watch -m1 frozen_opponent_ppo -m2 ceia_baseline_agent
"""

import argparse
import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
NUM_WORKERS = 8

BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent",
    "ray_results",
    "PPO_selfplay_twos",
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02",
    "checkpoint_002449",
    "checkpoint-2449",
)


def _load_baseline_weights():
    """Extract the baseline policy weights so we can seed a frozen policy."""
    config_dir = os.path.dirname(BASELINE_CHECKPOINT)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    with open(config_path, "rb") as f:
        baseline_config = pickle.load(f)

    baseline_config["num_workers"] = 0
    baseline_config["num_gpus"] = 0
    baseline_config.pop("callbacks", None)
    tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
    baseline_config["env"] = "DummyEnv"

    cls = get_trainable_cls("PPO")
    trainer = cls(env="DummyEnv", config=baseline_config)
    trainer.restore(BASELINE_CHECKPOINT)
    weights = trainer.get_weights(["default"])["default"]
    trainer.stop()
    return weights


def policy_mapping_fn(agent_id, *args, **kwargs):
    # Team 0 = learner; team 1 = mix of frozen baseline + lagged learner snapshots.
    if agent_id in (0, 1):
        return "default"
    return np.random.choice(
        ["baseline", "lagged_1", "lagged_2"],
        p=[0.60, 0.25, 0.15],
    )


class LaggedSnapshotCallback(DefaultCallbacks):
    """Periodically copy the live learner into lagged_* slots to diversify opponents."""

    def on_train_result(self, **info):
        result = info["result"]
        iteration = result.get("training_iteration", 0)
        if iteration > 0 and iteration % 25 == 0:
            trainer = info["trainer"]
            trainer.set_weights(
                {"lagged_2": trainer.get_weights(["lagged_1"])["lagged_1"]}
            )
            trainer.set_weights(
                {"lagged_1": trainer.get_weights(["default"])["default"]}
            )
            print(f"[iter {iteration}] Refreshed lagged opponents from learner.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--num-envs-per-worker", type=int, default=NUM_ENVS_PER_WORKER)
    args = parser.parse_args()
    NUM_WORKERS = args.num_workers
    NUM_ENVS_PER_WORKER = args.num_envs_per_worker

    ray.init(include_dashboard=True, dashboard_host="127.0.0.1")

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    trainer_gpus = 1 if available_gpus > 0 else 0
    print(f"Ray detected {available_gpus} GPU(s). Using num_gpus={trainer_gpus}.")

    print(f"Loading baseline weights from {BASELINE_CHECKPOINT} ...")
    baseline_weights = _load_baseline_weights()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    model_cfg = {
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }

    class SeedBaselineCallback(LaggedSnapshotCallback):
        def on_algorithm_init(self, *, algorithm, **kwargs):
            algorithm.set_weights({"baseline": baseline_weights})

        def on_train_result(self, **info):
            trainer = info["trainer"]
            iteration = info["result"].get("training_iteration", 0)
            if iteration == 1:
                trainer.set_weights({"baseline": baseline_weights})
                trainer.set_weights(
                    {"lagged_1": trainer.get_weights(["default"])["default"]}
                )
                trainer.set_weights(
                    {"lagged_2": trainer.get_weights(["default"])["default"]}
                )
            super().on_train_result(**info)

    analysis = tune.run(
        "PPO",
        name="PPO_vs_baseline",
        config={
            "num_gpus": trainer_gpus,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SeedBaselineCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "baseline": (None, obs_space, act_space, {}),
                    "lagged_1": (None, obs_space, act_space, {}),
                    "lagged_2": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": model_cfg,
            "lr": 3e-4,
            "gamma": 0.995,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 4096,
            "train_batch_size": 32768,
            "rollout_fragment_length": 1024,
            "batch_mode": "complete_episodes",
        },
        stop={"time_total_s": 21600},  # 6h cap; stop earlier with Ctrl-C
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(f"Best checkpoint: {best_checkpoint}")
    print("Done training")
