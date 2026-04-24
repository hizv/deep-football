"""Train agent_d_shaped: PPO with goal-aware PBRS shaping (FCNet policy).

One shared PPO policy applied to all four agents. Each env step produces
4x on-policy experience for the same network, which is sample-efficient and
sidesteps the stability headaches of archived self-play.

All continuous shaping is provided by GoalAwarePBRSWrapper; see reward_wrapper.py
for the design rationale.
"""

import argparse
import os
import sys

import gym
import ray
from ray import tune
from ray.rllib import MultiAgentEnv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from unity_compat import apply_unity_compat  # noqa: E402

apply_unity_compat()

import soccer_twos  # noqa: E402

from reward_wrapper import GoalAwarePBRSWrapper  # noqa: E402


class _RLlibMultiAgentEnv(gym.core.Wrapper, MultiAgentEnv):
    """Lets RLlib treat the base soccer_twos env as a MultiAgentEnv."""


def build_env(env_config=None):
    apply_unity_compat()

    raw_env_config = env_config or {}
    worker_index = getattr(raw_env_config, "worker_index", None)
    vector_index = getattr(raw_env_config, "vector_index", 0)

    env_config = dict(raw_env_config)
    if worker_index is not None:
        env_config["worker_id"] = (
            worker_index * env_config.get("num_envs_per_worker", 1)
            + vector_index
        )

    pbrs_gamma = float(env_config.pop("pbrs_gamma", 0.99))
    base_env = soccer_twos.make(**env_config)
    return GoalAwarePBRSWrapper(
        _RLlibMultiAgentEnv(base_env),
        pbrs_gamma=pbrs_gamma,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--time-total-s", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--keep-checkpoints-num", type=int, default=5)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentD_GoalAwarePBRS")
    return parser.parse_args()


def policy_mapping_fn(*_args, **_kwargs):
    return "default"


if __name__ == "__main__":
    args = parse_args()
    ray.init()

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    trainer_gpus = 1 if available_gpus > 0 else 0
    print(f"Ray detected {available_gpus} GPU(s). Using num_gpus={trainer_gpus}.")

    tune.registry.register_env("SoccerPBRS", build_env)

    probe_env = build_env(
        {
            "num_envs_per_worker": args.num_envs_per_worker,
            "pbrs_gamma": args.gamma,
        }
    )
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    probe_env.close()

    stop_criteria = {"timesteps_total": args.timesteps}
    if args.time_total_s > 0:
        stop_criteria["time_total_s"] = args.time_total_s

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": trainer_gpus,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "framework": "torch",
            "log_level": "INFO",
            # PPO — standard defaults with a slightly larger batch for the
            # multiagent rollout density.
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": 0.5,
            "rollout_fragment_length": 500,
            "train_batch_size": 8000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
            "env": "SoccerPBRS",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "pbrs_gamma": args.gamma,
            },
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
        stop=stop_criteria,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        keep_checkpoints_num=args.keep_checkpoints_num,
        local_dir=args.local_dir,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(f"Best checkpoint: {best_checkpoint}")
    print("Done training agent_d_shaped")
