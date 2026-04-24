"""Train agent_e_lstm_shaped: PPO with LSTM trunk + ray-based reward shaping.

Same reward wrapper as agent_d_shaped; difference is the policy network. LSTM
gives the policy memory across timesteps, which helps because:
  * The ray observation is egocentric and can lose sight of the ball for
    several frames during contests. Memory of "where was the ball going last
    tick" lets the agent keep pursuing it.
  * Kicks / possessions are multi-step events — the LSTM can learn a short
    trajectory prior rather than reacting purely to the current frame.

Gamma is bumped to 0.995 to avoid discounting sparse goal rewards too harshly
over the longer horizons an LSTM can plan across.
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
    parser = argparse.ArgumentParser(description="agent_e_lstm_shaped training")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--lstm-cell-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentE_LSTM_Shaped")
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
            "rollout_fragment_length": 1000,
            "train_batch_size": 12000,
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 5,
            "batch_mode": "truncate_episodes",
            "env": "SoccerShaped",
            "env_config": {"num_envs_per_worker": args.num_envs_per_worker},
            "model": {
                "use_lstm": True,
                "lstm_cell_size": args.lstm_cell_size,
                "max_seq_len": args.max_seq_len,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
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
    print("Done training agent_e_lstm_shaped")
