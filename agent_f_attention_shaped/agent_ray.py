"""Eval-time loader for agent_f_attention_shaped (DEBUG MODE)."""

import glob
import os
import pickle
import socket
from typing import Dict, List

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface

ALGORITHM = "PPO"
POLICY_NAME = "default"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_latest_checkpoint() -> str:
    results_dir = os.path.join(_THIS_DIR, "ray_results")
    candidates = glob.glob(os.path.join(results_dir, "**", "checkpoint-*"), recursive=True)
    candidates = [c for c in candidates if not c.endswith(".tune_metadata")]
    return max(candidates, key=lambda p: int(os.path.basename(p).split("-")[-1]))

def _get_free_port():
    """Finds a guaranteed free port to prevent Redis \x00 collisions."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class ProperDummyEnv(MultiAgentEnv):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self):
        return {}

    def step(self, action_dict):
        return {}, {}, {}, {}

class RayAgent(AgentInterface):
    """PPO agent with attention (GTrXL) and dense ray-based reward shaping."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "agent_f_attention_shaped"
        
        # --- THE REDIS FIX ---
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True, 
                include_dashboard=False, 
                log_to_driver=False,
                local_mode=True,  # Keep False so Unity sockets don't deadlock
                num_cpus=1
            )

        checkpoint_path = _find_latest_checkpoint()
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl") if os.path.exists(os.path.join(config_dir, "params.pkl")) else os.path.join(config_dir, "..", "params.pkl")
            
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # --- THE PICKLE FIX ---
        obs_space = env.observation_space
        act_space = env.action_space
        tune.registry.register_env(
            "ProperDummyEnv", 
            lambda env_config, o=obs_space, a=act_space: ProperDummyEnv(o, a)
        )
        config["env"] = "ProperDummyEnv"

        # --- THE GRADESCOPE MEMORY FIX ---
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["num_envs_per_worker"] = 1
        config["train_batch_size"] = 1
        config["rollout_fragment_length"] = 1
        config["sgd_minibatch_size"] = 1
        config["explore"] = False

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(checkpoint_path)
        self.policy = agent.get_policy(POLICY_NAME)
        
        # --- THE GTrXL EMPTY STATE FIX ---
        model_cfg = config.get("model", {})
        att_dim = model_cfg.get("attention_dim", 256)
        mem_inf = model_cfg.get("attention_memory_inference", 50)
        num_units = model_cfg.get("attention_num_transformer_units", 1)
        
        self._initial_state = [
            np.zeros((mem_inf, att_dim), dtype=np.float32)
            for _ in range(num_units)
        ]
        
        self.player_states: Dict[int, List[np.ndarray]] = {}
        self._step_counter = 0

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        self._step_counter += 1
        
        should_debug = (self._step_counter <= 2)
        
        if should_debug:
            print(f"\n{'='*40}\n[TICK {self._step_counter}] ENTERING ACT()")

        for player_id, obs in observation.items():
            if player_id not in self.player_states:
                self.player_states[player_id] = [
                    s.copy() for s in self._initial_state
                ]
            
            if should_debug:
                print(f" -> Player {player_id} Pre-Eval Memory: {[s.shape for s in self.player_states[player_id]]}")
            
            action, state_out, _ = self.policy.compute_single_action(
                obs,
                state=self.player_states[player_id],
                explore=False,
            )
            
            if should_debug:
                print(f" -> Player {player_id} Raw Action Shape: {np.array(action).shape}")
                print(f" -> Player {player_id} Output Memory Shape: {[s.shape for s in state_out]}")
            
            # --- THE SLIDING WINDOW FIX ---
            new_sliding_window = []
            for i in range(len(self.player_states[player_id])):
                old_mem_buffer = self.player_states[player_id][i]
                new_mem_frame = np.reshape(state_out[i], (1, -1))
                updated_buffer = np.concatenate([old_mem_buffer[1:], new_mem_frame], axis=0)
                new_sliding_window.append(updated_buffer)
                
            self.player_states[player_id] = new_sliding_window
            actions[player_id] = action
            
            if should_debug:
                print(f" -> Player {player_id} Post-Slide Window: {[s.shape for s in self.player_states[player_id]]}")
                
        if should_debug:
            print("="*40)
            
        return actions

    def reset(self):
        self.player_states.clear()
        self._step_counter = 0