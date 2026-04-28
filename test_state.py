import ray
from ray.tune.registry import register_env
from ray.rllib.env.base_env import BaseEnv
import glob
import pickle
import os

ray.init(ignore_reinit_error=True)
register_env('DummyEnv', lambda *_: BaseEnv())

def _find_latest_checkpoint() -> str:
    results_dir = "agent_f_attention_shaped/ray_results"
    candidates = glob.glob(os.path.join(results_dir, "**", "checkpoint-*"), recursive=True)
    candidates = [c for c in candidates if not c.endswith(".tune_metadata")]
    def _iter(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1
    return max(candidates, key=_iter)

checkpoint_path = _find_latest_checkpoint()
config_dir = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_dir, "params.pkl")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "..", "params.pkl")

with open(config_path, "rb") as f:
    p = pickle.load(f)
p['num_workers']=0
p['num_gpus']=0
from ray.rllib.agents.ppo import PPOTrainer
t = PPOTrainer(config=p, env='DummyEnv')
t.restore(checkpoint_path)

policy = t.get_policy("default")
print("Policy get_initial_state():", policy.get_initial_state())
print("Model get_initial_state():", policy.model.get_initial_state())

for k, v in policy.model.view_requirements.items():
    if "state" in k:
        print(f"View requirement {k}: space={v.space}, shape={v.space.shape if v.space else None}")
