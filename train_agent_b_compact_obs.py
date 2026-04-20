import argparse

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent B (compact kinematic observations, sparse reward)."
    )
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--experiment-name", type=str, default="AgentB_CompactObs")
    parser.add_argument("--position-scale", type=float, default=20.0)
    parser.add_argument("--velocity-scale", type=float, default=10.0)
    parser.add_argument("--distance-scale", type=float, default=30.0)
    return parser.parse_args()


def policy_mapping_fn(*_args, **_kwargs):
    return "default"


if __name__ == "__main__":
    args = parse_args()
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    env_config = {
        "variation": EnvType.multiagent_player,
        "num_envs_per_worker": args.num_envs_per_worker,
        "use_compact_obs": True,
        "return_dict_obs": False,
        "use_pbrs": False,
        "position_scale": args.position_scale,
        "velocity_scale": args.velocity_scale,
        "distance_scale": args.distance_scale,
    }
    temp_env = create_rllib_env(env_config)
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
            "env": "Soccer",
            "env_config": env_config,
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
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )
        print(best_checkpoint)
    print("Done training Agent B")
