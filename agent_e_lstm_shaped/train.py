"""Train agent_e_lstm_shaped: PPO + LSTM on goal-aware PBRS shaping.

Same shaping as agent_d_shaped; the model grows an LSTM trunk so the policy
can integrate information across timesteps.

Motivation for the LSTM:
  * The ray observation is egocentric and can briefly lose sight of the ball
    (occlusion by opponents, ball behind the agent). A recurrent state lets
    the policy remember "where the ball was going" for a few ticks rather
    than having to re-acquire from scratch.
  * Kicks and passes span multiple frames. An LSTM can learn a short
    trajectory prior (e.g., "ball is drifting toward me — prepare to strike")
    that a feedforward net cannot represent at all.

Hyperparameter adjustments relative to the FCNet variant:
  - lr lowered to 5e-5. LSTMs are higher-variance to gradient updates than
    FCNets; a more conservative lr reduces the risk of recurrent collapse.
  - rollout_fragment_length tracks max_seq_len so every collected chunk is a
    clean BPTT window.
  - recurrent PPO updates are gentler: larger train/minibatch sizes, fewer
    SGD passes, gradient clipping, and KL regularization.
  - vf_share_layers is disabled so the critic does not destabilize the policy
    trunk while its loss is still settling.
"""

import argparse
import os
import sys
from typing import Tuple

import gym
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib import MultiAgentEnv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_THIS_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from unity_compat import apply_unity_compat  # noqa: E402
from unity_ports import choose_base_port, find_free_base_port  # noqa: E402

apply_unity_compat()

import soccer_twos  # noqa: E402

from reward_wrapper import GoalAwarePBRSWrapper  # noqa: E402


class _RLlibMultiAgentEnv(gym.core.Wrapper, MultiAgentEnv):
    pass


class _RayCompatCallbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        apply_unity_compat()


def build_env(env_config=None):
    apply_unity_compat()

    raw_env_config = env_config or {}
    worker_index = getattr(raw_env_config, "worker_index", None)
    vector_index = getattr(raw_env_config, "vector_index", 0)

    env_config = dict(raw_env_config)
    if worker_index is not None:
        effective_worker_index = worker_index
        if env_config.get("driver_env_disabled", False) and worker_index > 0:
            effective_worker_index -= 1
        env_config["worker_id"] = (
            effective_worker_index * env_config.get("num_envs_per_worker", 1)
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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.02)
    parser.add_argument("--train-batch-size", type=int, default=16_000)
    parser.add_argument("--num-sgd-iter", type=int, default=4)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--kl-coeff", type=float, default=0.2)
    parser.add_argument("--kl-target", type=float, default=0.01)
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help=(
            "Unity ML-Agents base port. Defaults to a deterministic high port "
            "derived from the current user and Slurm job id."
        ),
    )
    parser.add_argument(
        "--port-search-limit",
        type=int,
        default=4096,
        help=(
            "Optional number of additional base ports to probe after selecting "
            "the initial base port. Defaults to 4096 to avoid collisions on "
            "shared nodes."
        ),
    )
    parser.add_argument("--lstm-cell-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--keep-checkpoints-num", type=int, default=5)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentE_LSTM_StablePBRS")
    return parser.parse_args()


def policy_mapping_fn(*_args, **_kwargs):
    return "default"


def _trainer_port_span(num_workers: int, num_envs_per_worker: int) -> int:
    # With create_env_on_driver disabled, only remote workers need Unity slots.
    # Keep one extra slot for the temporary probe env used to read spaces.
    return max(1, num_workers) * num_envs_per_worker + 1


def _available_cpu_budget() -> Tuple[int, str]:
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            return max(1, int(slurm_cpus_per_task)), "SLURM_CPUS_PER_TASK"
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0))), "os.sched_getaffinity(0)"
        except OSError:
            pass

    return max(1, int(ray.cluster_resources().get("CPU", 1))), "Ray cluster_resources"


def _effective_num_workers(requested_num_workers: int) -> int:
    available_cpus, cpu_source = _available_cpu_budget()
    max_remote_workers = max(0, available_cpus - 1)
    effective_num_workers = min(requested_num_workers, max_remote_workers)

    if effective_num_workers < requested_num_workers:
        print(
            "Detected "
            f"{available_cpus} CPU(s); reducing num_workers from "
            f"{requested_num_workers} to {effective_num_workers} "
            f"based on {cpu_source}."
        )

    return effective_num_workers


def _compatible_train_batch_size(
    requested_train_batch_size: int,
    num_workers: int,
    num_envs_per_worker: int,
    rollout_fragment_length: int,
) -> int:
    active_worker_count = max(1, num_workers)
    batch_unit = active_worker_count * num_envs_per_worker * rollout_fragment_length
    compatible_train_batch_size = max(
        batch_unit,
        (requested_train_batch_size // batch_unit) * batch_unit,
    )
    if compatible_train_batch_size != requested_train_batch_size:
        print(
            "Adjusting train_batch_size from "
            f"{requested_train_batch_size} to {compatible_train_batch_size} "
            f"so it is divisible by num_workers * num_envs_per_worker * "
            f"rollout_fragment_length = {batch_unit}."
        )
    return compatible_train_batch_size


if __name__ == "__main__":
    args = parse_args()
    ray.init()

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    trainer_gpus = 1 if available_gpus > 0 else 0
    print(f"Ray detected {available_gpus} GPU(s). Using num_gpus={trainer_gpus}.")

    tune.registry.register_env("SoccerPBRS", build_env)

    effective_num_workers = _effective_num_workers(args.num_workers)
    driver_env_disabled = effective_num_workers > 0
    reserved_ports = _trainer_port_span(
        effective_num_workers,
        args.num_envs_per_worker,
    )
    base_port = choose_base_port(args.base_port, reserved_ports)
    if args.port_search_limit > 0:
        base_port = find_free_base_port(
            base_port,
            reserved_ports,
            search_limit=args.port_search_limit,
        )
    probe_worker_id = reserved_ports - 1
    print(
        "Using Unity base_port="
        f"{base_port} with {reserved_ports} reserved worker slots "
        f"across {effective_num_workers} Ray worker(s)."
    )

    probe_env = build_env(
        {
            "num_envs_per_worker": args.num_envs_per_worker,
            "pbrs_gamma": args.gamma,
            "base_port": base_port,
            "worker_id": probe_worker_id,
        }
    )
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    probe_env.close()

    # Keep sgd_minibatch_size a multiple of max_seq_len so LSTM BPTT windows
    # pack cleanly during SGD while averaging across enough sequences to keep
    # recurrent gradients from getting too noisy.
    sgd_minibatch_size = max(args.max_seq_len * 40, 800)
    train_batch_size = _compatible_train_batch_size(
        args.train_batch_size,
        effective_num_workers,
        args.num_envs_per_worker,
        args.max_seq_len,
    )

    stop_criteria = {"timesteps_total": args.timesteps}
    if args.time_total_s > 0:
        stop_criteria["time_total_s"] = args.time_total_s

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config={
            "num_gpus": trainer_gpus,
            "num_workers": effective_num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "create_env_on_driver": False,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": _RayCompatCallbacks,
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": args.vf_loss_coeff,
            "grad_clip": args.grad_clip,
            "kl_coeff": args.kl_coeff,
            "kl_target": args.kl_target,
            "rollout_fragment_length": args.max_seq_len,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": args.num_sgd_iter,
            "batch_mode": "truncate_episodes",
            "env": "SoccerPBRS",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "pbrs_gamma": args.gamma,
                "base_port": base_port,
                "driver_env_disabled": driver_env_disabled,
            },
            "model": {
                "use_lstm": True,
                "lstm_cell_size": args.lstm_cell_size,
                "max_seq_len": args.max_seq_len,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
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
    print("Done training agent_e_lstm_shaped")
