import argparse
import numpy as np
import ray
import yaml
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from soccer_twos import EnvType

from mappo_model import MAPPOCentralCriticModel
from utils import create_rllib_env, sample_player, sample_pos_vel


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Agent C with MAPPO-style centralized critic, compact observations, "
            "PBRS, curriculum, and archived self-play snapshots."
        )
    )
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--time-limit-s", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--experiment-name", type=str, default="AgentC_MAPPO")
    parser.add_argument("--curriculum", type=str, default="curriculum.yaml")
    parser.add_argument("--curriculum-threshold", type=float, default=1.0)
    parser.add_argument("--snapshot-threshold", type=float, default=0.6)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--pbrs-alpha", type=float, default=1.0)
    parser.add_argument("--pbrs-beta", type=float, default=0.3)
    parser.add_argument("--pbrs-scale", type=float, default=1.0)
    parser.add_argument("--position-scale", type=float, default=20.0)
    parser.add_argument("--velocity-scale", type=float, default=10.0)
    parser.add_argument("--distance-scale", type=float, default=30.0)
    return parser.parse_args()


def load_curriculum(curriculum_path):
    with open(curriculum_path) as file_handle:
        data = yaml.load(file_handle, Loader=yaml.FullLoader)
    tasks = data.get("tasks", [])
    if not tasks:
        raise ValueError("Curriculum file must include at least one task")
    return tasks


def make_callbacks(tasks, curriculum_threshold, snapshot_threshold, snapshot_interval):
    config_fns = {
        "none": lambda *_args, **_kwargs: None,
        "random_players": lambda env: env.set_policies(
            lambda *_args, **_kwargs: env.action_space.sample()
        ),
    }

    class CurriculumSnapshotCallback(DefaultCallbacks):
        current_task = 0
        next_snapshot_slot = 1
        last_snapshot_iteration = -10_000

        def on_episode_start(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            env_index,
            **kwargs,
        ):
            task = tasks[self.__class__.current_task]

            for env in base_env.get_unwrapped():
                config_fn = config_fns.get(task.get("config_fn", "none"), config_fns["none"])
                try:
                    config_fn(env)
                except Exception:
                    # Training should continue even if optional policy randomization is unavailable.
                    pass

                ranges = task.get("ranges", {})
                ball_ranges = ranges.get("ball", {})
                player_ranges = ranges.get("players", {})
                try:
                    env.env_channel.set_parameters(
                        ball_state=sample_pos_vel(ball_ranges),
                        players_states={
                            int(player_id): sample_player(player_ranges[player_id])
                            for player_id in player_ranges
                        },
                    )
                except Exception:
                    # Some environment variants may not expose the configuration side channel.
                    pass

        def on_train_result(self, **info):
            result = info["result"]
            trainer = info["trainer"]
            training_iteration = int(result.get("training_iteration", 0))
            mean_reward = float(result.get("episode_reward_mean", 0.0))

            if (
                mean_reward >= curriculum_threshold
                and self.__class__.current_task < len(tasks) - 1
            ):
                self.__class__.current_task += 1
                print(
                    "---- Curriculum update -> stage "
                    f"{self.__class__.current_task}: "
                    f"{tasks[self.__class__.current_task]['name']} ----"
                )

            if (
                mean_reward >= snapshot_threshold
                and (training_iteration - self.__class__.last_snapshot_iteration)
                >= snapshot_interval
            ):
                slot = self.__class__.next_snapshot_slot
                target_policy = f"opponent_{slot}"
                local_policy_map = trainer.workers.local_worker().policy_map
                if target_policy in local_policy_map:
                    main_weights = trainer.get_weights(["main"])["main"]
                    trainer.set_weights({target_policy: main_weights})
                    self.__class__.last_snapshot_iteration = training_iteration
                    self.__class__.next_snapshot_slot = 1 + (slot % 3)
                    print(f"---- Snapshot updated: {target_policy} ----")

            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["curriculum_stage"] = self.__class__.current_task
            result["custom_metrics"]["snapshot_slot"] = self.__class__.next_snapshot_slot

    return CurriculumSnapshotCallback


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id in (0, 1):
        return "main"

    candidate_policies = ["main", "opponent_1", "opponent_2", "opponent_3"]
    candidate_probs = [0.50, 0.25, 0.15, 0.10]

    if episode is not None:
        if "opponent_policy" not in episode.user_data:
            episode.user_data["opponent_policy"] = np.random.choice(
                candidate_policies, p=candidate_probs
            )
        return episode.user_data["opponent_policy"]

    return np.random.choice(candidate_policies, p=candidate_probs)


if __name__ == "__main__":
    args = parse_args()
    tasks = load_curriculum(args.curriculum)

    ray.init()
    tune.registry.register_env("Soccer", create_rllib_env)
    ModelCatalog.register_custom_model("mappo_central_critic", MAPPOCentralCriticModel)

    env_config = {
        "variation": EnvType.multiagent_player,
        "num_envs_per_worker": args.num_envs_per_worker,
        "use_compact_obs": True,
        "return_dict_obs": True,
        "use_pbrs": True,
        "pbrs_alpha": args.pbrs_alpha,
        "pbrs_beta": args.pbrs_beta,
        "pbrs_gamma": args.gamma,
        "pbrs_scale": args.pbrs_scale,
        "position_scale": args.position_scale,
        "velocity_scale": args.velocity_scale,
        "distance_scale": args.distance_scale,
    }

    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    callbacks_cls = make_callbacks(
        tasks=tasks,
        curriculum_threshold=args.curriculum_threshold,
        snapshot_threshold=args.snapshot_threshold,
        snapshot_interval=args.snapshot_interval,
    )

    stop_config = {"timesteps_total": args.timesteps}
    if args.time_limit_s > 0:
        stop_config["time_total_s"] = args.time_limit_s

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": int(ray.cluster_resources().get("GPU", 0) > 0),
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": callbacks_cls,
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 10,
            "env": "Soccer",
            "env_config": env_config,
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
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["main"],
            },
        },
        stop=stop_config,
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
    print("Done training Agent C")
