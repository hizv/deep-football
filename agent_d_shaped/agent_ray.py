"""Eval-time loader for agent_d_shaped.

Restores the trained PPO policy and applies it independently to each of the
two team members. Mirrors the loader pattern in DRL-soccer-playing/MY_AGENT.
"""

import glob
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_latest_checkpoint() -> str:
    """Return path to the highest-numbered checkpoint-<N> file under ray_results/."""
    results_dir = os.path.join(_THIS_DIR, "ray_results")
    candidates = glob.glob(os.path.join(results_dir, "**", "checkpoint-*"), recursive=True)
    candidates = [c for c in candidates if not c.endswith(".tune_metadata")]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {results_dir}. Run train.py first."
        )

    def _iter(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    return max(candidates, key=_iter)


class RayAgent(AgentInterface):
    """PPO agent with dense ray-based reward shaping."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "agent_d_shaped"
        ray.init(ignore_reinit_error=True)

        checkpoint_path = _find_latest_checkpoint()
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "..", "params.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(checkpoint_path)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            actions[player_id], *_ = self.policy.compute_single_action(obs, explore=False)
        return actions
