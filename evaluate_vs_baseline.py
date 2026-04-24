import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DISABLE_METRICS_COLLECTION", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_DISABLE_REPORTER", "1")

from soccer_twos.evaluate import evaluate


def _json_default(obj):
    # soccer_twos evaluator returns NumPy scalar/array values in nested dicts.
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Evaluate two SoccerTwos agent modules.")
    parser.add_argument("--agent1", type=str, required=True, help="Module name for agent 1")
    parser.add_argument("--agent2", type=str, required=True, help="Module name for agent 2")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--base-port", type=int, default=None, help="Optional base port")
    args = parser.parse_args()

    default_base_port = 15000 + (os.getpid() % 40000)
    base_port = args.base_port if args.base_port is not None else default_base_port

    print(f"[Config] base_port={base_port}")

    result = evaluate(
        agent1_module_name=args.agent1,
        agent2_module_name=args.agent2,
        n_episodes=args.episodes,
        base_port=base_port,
    )

    p1 = result["policies"][args.agent1]
    p2 = result["policies"][args.agent2]

    print("===== HEAD TO HEAD SUMMARY =====")
    print(f"Agent 1: {args.agent1}")
    print(f"  wins/losses/draws: {p1['policy_wins']}/{p1['policy_losses']}/{p1['policy_draws']}")
    print(f"  win rate: {p1['policy_win_rate']:.2%}")
    print(f"  reward mean: {p1['policy_reward_mean']:.4f}")
    print()
    print(f"Agent 2: {args.agent2}")
    print(f"  wins/losses/draws: {p2['policy_wins']}/{p2['policy_losses']}/{p2['policy_draws']}")
    print(f"  win rate: {p2['policy_win_rate']:.2%}")
    print(f"  reward mean: {p2['policy_reward_mean']:.4f}")

    out_file = f"eval_{args.agent1}_vs_{args.agent2}_{args.episodes}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"\nFull result saved to: {out_file}")


if __name__ == "__main__":
    main()
