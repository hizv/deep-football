import os
import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface
from soccer_twos import EnvType

from mappo_model import MAPPOCentralCriticModel
from utils import create_rllib_env
from train_agent_c_mappo import register_model_safely

class AgentC(AgentInterface):
    """ MAPPO Agent wrapper for evaluation. """
    def __init__(self, env):
        self.action_space = env.action_space

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        ray.tune.registry.register_env("Soccer", create_rllib_env)
        register_model_safely("mappo_central_critic", MAPPOCentralCriticModel)

        # Same config used in train_agent_c_mappo.py
        env_config = {
            "variation": EnvType.multiagent_player,
            "use_compact_obs": True,
            "return_dict_obs": True,
            "use_pbrs": False,
            "base_port": 16000 + (os.getpid() % 10000), # use an isolated port
        }

        # Get observation and action spaces
        temp_env = create_rllib_env(env_config)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        temp_env.close()

        config = {
            "env": "Soccer",
            "env_config": env_config,
            "framework": "torch",
            "num_workers": 0,
            "model": {
                "custom_model": "mappo_central_critic",
                "vf_share_layers": False,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "custom_model_config": {
                    "critic_hiddens": [256, 256, 256],
                },
            },
            "multiagent": {
                "policies": {
                    "main": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "main",
            },
            "explore": False, # Ensure deterministic actions during evaluation
        }
        
        self.trainer = PPOTrainer(config=config)
        
        # Load the latest checkpoint
        # Make sure this matches your actual best checkpoint!
        checkpoint_path = os.path.abspath(
            "ray_results/AgentC_MAPPO/PPO_Soccer_1b873_00000_0_2026-04-22_10-03-58/checkpoint_000330/checkpoint-330"
        )
        self.trainer.restore(checkpoint_path)

    def act(self, observation):
        actions = {}
        for player_id, obs in observation.items():
            # Get deterministic action from the central policy
            action = self.trainer.compute_action(
                obs, policy_id="main", explore=False
            )
            actions[player_id] = action
            
        return actions
