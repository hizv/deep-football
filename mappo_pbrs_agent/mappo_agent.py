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
from mappo_model import MAPPOCentralCriticModel

from ray.rllib.models import ModelCatalog
from train_agent_c_mappo import register_model_safely

# Set this to your specific MAPPO checkpoint directory
CHECKPOINT_PATH = os.environ.get(
    "MAPPO_CHECKPOINT",
    "./ray_results/MAPPO_vs_Random/checkpoint_000000/checkpoint-0"
)
ALGORITHM = "PPO"
POLICY_NAME = "default"


class MAPPO_Agent(AgentInterface):
    """
    The class MUST be named 'Agent' for the soccer_twos evaluator to detect it.
    """
    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(
                f"Could not find params.pkl near checkpoint: {CHECKPOINT_PATH}"
            )

        # Disable scaling and GPUs for evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # Register a dummy env to satisfy RLlib's initialization requirements
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"
        
        register_model_safely("mappo_central_critic", MAPPOCentralCriticModel)

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        
        self.policy = agent.get_policy(POLICY_NAME)
        
        # Track hidden states for Frame Stacking
        self.states = {}

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            
            # 1. Initialize frame stacking state for new players
            if player_id not in self.states:
                self.states[player_id] = self.policy.get_initial_state()

            # 2. Check for Observation Wrapper Mismatch
            # If your policy was trained with CompactObservationRewardWrapper, 
            # the raw 'obs' here (shape ~334) will crash the network (expects ~42).
            # You must apply your observation extraction logic here if needed.
            processed_obs = obs 

            # 3. Compute action and update hidden state
            action, state_out, _ = self.policy.compute_single_action(
                processed_obs,
                state=self.states[player_id],
                explore=False # Force deterministic actions for evaluation
            )
            
            actions[player_id] = action
            self.states[player_id] = state_out

        return actions