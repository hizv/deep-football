import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default"

# Path to the trained checkpoint. Set via env var FROZEN_OPPONENT_CHECKPOINT to
# override, otherwise the loader walks ray_results/ for the newest checkpoint
# under a run named "PPO_vs_baseline" or "PPO_selfplay".
_DEFAULT_CHECKPOINT_ENV = "FROZEN_OPPONENT_CHECKPOINT"
_RAY_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ray_results"
)
_PREFERRED_RUN_PREFIXES = ("PPO_vs_baseline", "PPO_selfplay", "PPO_SP", "PPO_")


def _find_latest_checkpoint() -> str:
    env_override = os.environ.get(_DEFAULT_CHECKPOINT_ENV)
    if env_override:
        return env_override

    local_results = os.path.abspath(_RAY_RESULTS_DIR)
    if not os.path.isdir(local_results):
        raise FileNotFoundError(
            f"No checkpoint found. Set env var {_DEFAULT_CHECKPOINT_ENV}=/path/to/checkpoint or "
            f"run train_ppo_vs_baseline.py to generate checkpoints under {local_results}."
        )

    candidates = []
    for run_name in os.listdir(local_results):
        if not run_name.startswith(_PREFERRED_RUN_PREFIXES):
            continue
        run_dir = os.path.join(local_results, run_name)
        if not os.path.isdir(run_dir):
            continue
        for trial_name in os.listdir(run_dir):
            trial_dir = os.path.join(run_dir, trial_name)
            if not os.path.isdir(trial_dir):
                continue
            for entry in os.listdir(trial_dir):
                if not entry.startswith("checkpoint_"):
                    continue
                ckpt_dir = os.path.join(trial_dir, entry)
                for f in os.listdir(ckpt_dir):
                    if f.startswith("checkpoint-") and not f.endswith(".tune_metadata"):
                        candidates.append(os.path.join(ckpt_dir, f))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {local_results}. Train one first with "
            "train_ppo_vs_baseline.py."
        )
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


class TeamAgent(AgentInterface):
    """PPO-based team agent trained against the baseline as a frozen opponent.

    Loads a Ray RLlib checkpoint and mirrors the ceia_baseline_agent loading pattern
    so that eval via `python -m soccer_twos.watch -m1 frozen_opponent_ppo -m2 ceia_baseline_agent`
    works out of the box.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        checkpoint_path = _find_latest_checkpoint()
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError(f"params.pkl not found near {checkpoint_path}")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        # Strip callbacks so we don't need the training module at eval time.
        config.pop("callbacks", None)

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        trainer.restore(checkpoint_path)
        self.policy = trainer.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
