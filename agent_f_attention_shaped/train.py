"""Train agent_f_gtrxl_shaped: PPO + GTrXL on goal-aware PBRS shaping.

Replaces the LSTM trunk with a Gated Transformer-XL (GTrXL). 
GTrXL uses GRU gating around the transformer blocks to stabilize RL training, 
preventing the high variance that usually plagues standard Transformers in RL.

Hyperparameter adjustments relative to the LSTM variant:
  - max_seq_len (now attention_memory_training) is carefully tuned. 
    Transformers are O(N^2) in memory; a window of 50 is usually plenty for physics.
  - attention_init_gru_gate_bias set to 2.0. This forces the transformer to 
    start acting like a standard feed-forward network, slowly opening the 
    attention gates as it learns, preventing catastrophic initial variance.
"""

import argparse
import os
import sys

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
from unity_ports import find_free_base_port  # noqa: E402

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
        env_config["worker_id"] = (
            worker_index * env_config.get("num_envs_per_worker", 1)
            + vector_index
        )

    pbrs_gamma = float(env_config.pop("pbrs_gamma", 0.99))
    base_env = soccer_twos.make(**env_config)
    
    # --- THE GTrXL SHAPING OVERRIDES ---
    # We pass the 10x scaled-down values here so the Transformer
    # isn't deafened by the dense shaping signals.
    return GoalAwarePBRSWrapper(
        _RLlibMultiAgentEnv(base_env),
        pbrs_gamma=pbrs_gamma,
        alpha=0.1,                # Down from 1.0
        beta=0.03,                # Down from 0.3
        kick_base=0.004,          # Down from 0.04
        kick_goal_bonus=0.006,    # Down from 0.06
        potential_scale=0.01      # Leave this at 0.01 since we shrank alpha/beta
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
    parser.add_argument("--base-port", type=int, default=50039)
    
    # --- Transformer Specific Args ---
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-memory", type=int, default=50) # The Transformer "seq_len"
    
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--keep-checkpoints-num", type=int, default=5)
    parser.add_argument("--local-dir", type=str, default=os.path.join(_THIS_DIR, "ray_results"))
    parser.add_argument("--experiment-name", type=str, default="AgentF_GTrXL_GoalAwarePBRS")
    return parser.parse_args()


def policy_mapping_fn(*_args, **_kwargs):
    return "default"


def _trainer_port_span(num_workers: int, num_envs_per_worker: int) -> int:
    return (num_workers + 1) * num_envs_per_worker + 1


if __name__ == "__main__":
    args = parse_args()
    ray.init(
        include_dashboard=False, 
        log_to_driver=False,
        ignore_reinit_error=True,
    )

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    trainer_gpus = 1 if available_gpus > 0 else 0
    print(f"Ray detected {available_gpus} GPU(s). Using num_gpus={trainer_gpus}.")

    tune.registry.register_env("SoccerPBRS", build_env)

    reserved_ports = _trainer_port_span(args.num_workers, args.num_envs_per_worker)
    base_port = find_free_base_port(args.base_port, reserved_ports)
    probe_worker_id = reserved_ports - 1
    print(
        f"Using Unity base_port={base_port} with {reserved_ports} reserved worker slots."
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

    # Keep sgd_minibatch_size a multiple of attention_memory so chunks pack cleanly
    sgd_minibatch_size = max(args.attention_memory * 10, 200)

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
            "callbacks": _RayCompatCallbacks,
            "lr": args.lr,
            "gamma": args.gamma,
            "lambda": args.gae_lambda,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": 0.5,
            
            # Use attention window for rollout fragments
            "rollout_fragment_length": args.attention_memory,
            "train_batch_size": 8000,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
            "env": "SoccerPBRS",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "pbrs_gamma": args.gamma,
                "base_port": base_port,
            },
            
            # --- THE GTrXL MODEL CONFIGURATION ---
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
                
                "use_lstm": False,
                "use_attention": True,
                
                "attention_num_transformer_units": 1,
                "attention_dim": args.attention_dim,
                "attention_num_heads": args.attention_heads,
                "attention_head_dim": args.attention_dim // args.attention_heads,
                "attention_memory_inference": args.attention_memory,
                "attention_memory_training": args.attention_memory,
                "attention_position_wise_mlp_dim": 256,
                "attention_init_gru_gate_bias": 2.0, # CRITICAL: Stabilizes early RL training
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
    print("Done training agent_f_gtrxl_shaped")