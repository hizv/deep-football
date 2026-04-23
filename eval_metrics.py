import argparse
import numpy as np
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
import soccer_twos
from soccer_twos import EnvType
import os

class RawObsWrapper(gym.core.Wrapper):
    def step(self, action):
        self.raw_obs, rewards, dones, infos = self.env.step(action)
        return self.raw_obs, rewards, dones, infos
    def reset(self, **kwargs):
        self.raw_obs = self.env.reset(**kwargs)
        return self.raw_obs

from mappo_model import MAPPOCentralCriticModel
from utils import create_rllib_env
from train_agent_c_mappo import register_model_safely
from example_team_agent.agent import TeamAgent

os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DISABLE_METRICS_COLLECTION", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_DISABLE_REPORTER", "1")

def evaluate(checkpoint_path, opponent_type="random", num_matches=10, base_port=None):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    ray.tune.registry.register_env("Soccer", create_rllib_env)
    register_model_safely("mappo_central_critic", MAPPOCentralCriticModel)

    env_config = {
        "variation": EnvType.multiagent_player,
        "use_compact_obs": True,
        "return_dict_obs": True,
    }
    if base_port is not None:
        env_config["base_port"] = base_port

    # Get spaces by briefly creating the env
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
            "custom_model_config": {"critic_hiddens": [256, 256, 256]},
        },
        "multiagent": {
            "policies": {
                "main": (None, obs_space, act_space, {}),
                "opponent_1": (None, obs_space, act_space, {}),
                "opponent_2": (None, obs_space, act_space, {}),
                "opponent_3": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id, **kwargs: "main",
        }
    }

    print(f"Loading checkpoint: {checkpoint_path}")
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    
    # Use a safely offset port for the evaluation environment
    eval_env_config = env_config.copy()
    if base_port is not None:
        eval_env_config["base_port"] = base_port + 50
    eval_env_config["render"] = False
        
    # Create the environment with our wrapper so RLLib gets dict observations
    from utils import CompactObservationRewardWrapper
    base_raw_env = soccer_twos.make(**eval_env_config)
    raw_env = RawObsWrapper(base_raw_env)
    env = CompactObservationRewardWrapper(raw_env, env_config)
    
    if opponent_type == "baseline":
        baseline_agent = TeamAgent(base_raw_env)
    
    wins = 0
    draws = 0
    losses = 0
    
    print(f"\nStarting {num_matches} matches against {opponent_type.upper()} opponent...")
    
    for match_id in range(1, num_matches + 1):
        obs = env.reset()
        done = False
        team_reward = 0
        
        while not done:
            actions = {}
            # Split observations between our MAPPO agent (Team 0: players 0, 1) 
            # and the opponent (Team 1: players 2, 3)
            
            # Our Agents (Players 0, 1)
            for agent_id in [0, 1]:
                if agent_id in obs:
                    action = trainer.compute_action(obs[agent_id], policy_id="main")
                    actions[agent_id] = action

            # Opponent Agents (Players 2, 3)
            if opponent_type == "random":
                for agent_id in [2, 3]:
                    if agent_id in obs:
                        # random action
                        actions[agent_id] = env.action_space.sample()
            
            elif opponent_type == "baseline":
                # The baseline agent expects a dict of the RAW observations
                opp_obs = {p: raw_env.raw_obs[p] for p in [2, 3] if p in raw_env.raw_obs}
                if opp_obs:
                    baseline_actions = baseline_agent.act(opp_obs)
                    actions.update(baseline_actions)
                    
            obs, rewards, dones, info = env.step(actions)
            
            # Accumulate reward from our team's perspective
            if 0 in rewards:
                team_reward += rewards[0]
            elif 1 in rewards:
                team_reward += rewards[1]
                
            if dones["__all__"]:
                done = True
                
        # soccer_twos typically returns standard +1 win, -1 loss, 0 draw
        if team_reward > 0:
            wins += 1
            result = "WIN"
        elif team_reward < 0:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
            
        print(f"Match {match_id}: {result} (Net Reward: {team_reward})")

    print("\n" + "="*40)
    print(f"FINAL RESULTS DRAFT VS {opponent_type.upper()}")
    print(f"Wins: {wins}/{num_matches}")
    print(f"Draws: {draws}/{num_matches}")
    print(f"Losses: {losses}/{num_matches}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--opponent", type=str, choices=["random", "baseline"], default="random")
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--base-port", type=int, default=15000)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.opponent, args.matches, args.base_port)
