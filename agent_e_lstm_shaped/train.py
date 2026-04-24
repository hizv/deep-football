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
  - lr lowered to 1e-4. LSTMs are higher-variance to gradient updates than
    FCNets; conservative lr reduces the risk of representational collapse in
    the recurrent state.
  - rollout_fragment_length set to 200 (equal to max_seq_len) so every
    collected chunk is a clean BPTT window.
  - sgd_minibatch_size is an integer multiple of max_seq_len so minibatches
    pack cleanly when SGD iterates.
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
    pass


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--lstm-cell-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--keep-checkpoints-num", type=int, default=5)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentE_LSTM_GoalAwarePBRS")
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

    # Keep sgd_minibatch_size a multiple of max_seq_len so LSTM BPTT windows
    # pack cleanly during SGD.
    sgd_minibatch_size = max(args.max_seq_len * 10, 200)

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
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": 0.5,
            "rollout_fragment_length": args.max_seq_len,
            "train_batch_size": 8000,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
            "env": "SoccerPBRS",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "pbrs_gamma": args.gamma,
            },
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
