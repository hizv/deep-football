import os
import pickle
import sys
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls, RLLIB_MODEL, _global_registry

from soccer_twos import AgentInterface

# Add project root so mappo_model can be imported regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mappo_model import MAPPOCentralCriticModel  # noqa: E402

ALGORITHM = "PPO"

# Best checkpoint — 334 iterations, 10M timesteps. Stored inside this module
# directory so the agent is self-contained when packaged as a zip.
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/AgentC_MAPPO/"
    "PPO_Soccer_fbab6_00000_0_2026-04-22_11-36-08/"
    "checkpoint_000334/checkpoint-334",
)

POLICY_NAME = "main"


def _register_mappo_model():
    try:
        ModelCatalog.register_custom_model("mappo_central_critic", MAPPOCentralCriticModel)
    except AttributeError as err:
        if "keras" not in str(err):
            raise
        _global_registry.register(RLLIB_MODEL, "mappo_central_critic", MAPPOCentralCriticModel)


class RayAgent(AgentInterface):
    """Agent C — MAPPO centralized critic, compact observations, PBRS, curriculum + self-play."""

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)
        _register_mappo_model()

        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"params.pkl not found near {CHECKPOINT_PATH}"
            )

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            # The MAPPO policy was trained with dict obs {"obs": ..., "state": ...}.
            # At eval time soccer_twos passes raw flat observations, so we wrap
            # them to match the expected input format.
            if not isinstance(obs, dict):
                obs = {"obs": obs, "state": obs}
            actions[player_id], *_ = self.policy.compute_single_action(obs)
        return actions
