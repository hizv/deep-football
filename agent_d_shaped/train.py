"""Train agent_d_shaped: PPO + dense ray-based reward shaping.

Single shared PPO policy across all 4 agents (one network, applied to both
teams). This gives stable training without the overhead of archived self-play:
every step, all 4 agents produce on-policy experience for the same network.

Dense rewards come from the ray-based wrapper (proximity / progress /
possession / kick / spread). No goal-info or ball-info fields required —
everything is computed from the raw 42x8 ray observation.
"""

import argparse
import os
import sys

import ray
from ray import tune
from ray.rllib import MultiAgentEnv
import gym
import soccer_twos

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reward_wrapper import RayBasedRewardWrapper  # noqa: E402


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_env(env_config=None):
    env_config = dict(env_config or {})
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    return RayBasedRewardWrapper(RLLibWrapper(env))


def parse_args():
    parser = argparse.ArgumentParser(description="agent_d_shaped training")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentD_Shaped")
    return parser.parse_args()


def policy_mapping_fn(*_args, **_kwargs):
    return "default"


if __name__ == "__main__":
    args = parse_args()
    ray.init()

    tune.registry.register_env("SoccerShaped", create_env)

    temp_env = create_env({"num_envs_per_worker": args.num_envs_per_worker})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": int(ray.cluster_resources().get("GPU", 0) > 0),
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "framework": "torch",
            "log_level": "INFO",
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": 0.5,
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
            "env": "SoccerShaped",
            "env_config": {"num_envs_per_worker": args.num_envs_per_worker},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
        },
        stop={"timesteps_total": args.timesteps},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        keep_checkpoints_num=5,
        local_dir=args.local_dir,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is not None:
        print(analysis.get_best_checkpoint(best_trial, "episode_reward_mean", "max"))
    print("Done training agent_d_shaped")
