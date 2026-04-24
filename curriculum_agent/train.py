import ray
import yaml
import os
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.socc import soccer_twos 

# --- YAML LOADER ---
with open(os.getenv("CURRICULUM_YAML_PATH", "curriculum.yaml"), "r") as f:
    CURRICULUM_DATA = yaml.safe_load(f)
    TASKS = CURRICULUM_DATA["tasks"]

# --- WRAPPER TO HANDLE YAML RANGES ---
class SoccerCurriculumWrapper(SoccerEnvWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.current_task_idx = 0

    def reset(self):
        # Get the ranges for the current task level
        task_config = TASKS[self.current_task_idx].get("ranges", {})
        # Inject ranges into Unity reset
        return self.env.reset(config=task_config)

    def set_task(self, task_idx):
        self.current_task_idx = task_idx

    def get_task(self):
        return self.current_task_idx

# --- CALLBACK TO LEVEL UP ---
class CurriculumCallback(ray.rllib.agents.callbacks.DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        # Access task level from the local worker
        current_idx = trainer.workers.local_worker().env.get_task()
        
        # Difficulty Thresholds
        # Task 0 (Very Easy) -> Task 1 (Easy) at 0.8 reward
        # Task 4 (Random) is the final state
        threshold = 0.8 if current_idx < 2 else 0.5
        
        if result["episode_reward_mean"] > threshold and current_idx < len(TASKS) - 1:
            new_idx = current_idx + 1
            # Broadcast new level to all 14 workers
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(lambda env: env.set_task(new_idx))
            )
            print(f"\n[LEVEL UP] Completed: {TASKS[current_idx]['name']}")
            print(f"[LEVEL UP] Starting: {TASKS[new_idx]['name']}\n")

# --- MAIN TRAINING LOOP ---
def run():
    ray.init(ignore_reinit_error=True)
    
    config = {
        "env": SoccerCurriculumWrapper,
        "env_config": {
            "env_type": EnvType.COMPETITION, 
        },
        "framework": "torch",
        "num_workers": 14,
        "num_envs_per_worker": 2,
        "train_batch_size": 14000,
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 5e-5,
        "num_sgd_iter": 16,
        "callbacks": CurriculumCallback,
        "model": {
            "fcnet_hiddens": [512, 256, 128], # Matches your "Diet" tweak
        },
        "multiagent": {
            "policies": {"default_policy"},
            "policy_mapping_fn": lambda agent_id, **kwargs: "default_policy",
        },
    }

    tune.run(
        "PPO",
        config=config,
        stop={"timesteps_total": 40000000},
        checkpoint_freq=50,
        local_dir="./ray_results/curriculum_results",
    )

if __name__ == "__main__":
    run()