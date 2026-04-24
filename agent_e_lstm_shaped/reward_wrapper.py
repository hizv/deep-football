"""Ray-based dense reward shaping for soccer_twos.

Adapted from DRL-soccer-playing/A_SHOURIK_AGENT/reward_wrapper.py and
DRL-soccer-playing/MY_AGENT/my_utils.py. Uses only the raw 336-dim ray
observations (42 rays x 8 channels) so it works without info-dict fields.

Per-step shaping on top of the sparse goal reward:
  1. proximity  — closer to ball = more reward (saturating).
  2. progress   — reward when agent's ball distance drops vs last step.
  3. possession — bonus to the agent on each team closest to the ball.
  4. kick       — larger bonus when an agent was touching the ball and
                  it then flew away (dist jumped by > KICK_ESCAPE_MIN).
  5. spread     — bonus when teammate is far, discouraging ball-clumping.
"""

import gym
import numpy as np
from ray.rllib import MultiAgentEnv


RAY_SIZE = 8
NUM_RAYS = 42
BALL_TAG_IDX = 0
BLUE_AGENT_TAG_IDX = 3
PURPLE_AGENT_TAG_IDX = 4
DIST_IDX = 7

KICK_THRESHOLD = 0.15
KICK_ESCAPE_MIN = 0.10
SPREAD_THRESHOLD = 0.55


class RayBasedRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(
        self,
        env,
        ball_proximity_weight: float = 0.005,
        ball_progress_weight: float = 0.01,
        possession_weight: float = 0.002,
        kick_weight: float = 0.05,
        spread_weight: float = 0.003,
    ):
        super().__init__(env)
        self.ball_proximity_weight = ball_proximity_weight
        self.ball_progress_weight = ball_progress_weight
        self.possession_weight = possession_weight
        self.kick_weight = kick_weight
        self.spread_weight = spread_weight
        self._prev_ball_dist: dict = {}

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _min_tag_dist(self, obs: np.ndarray, tag_idx: int) -> float:
        min_dist = 1.0
        for i in range(NUM_RAYS):
            base = i * RAY_SIZE
            if obs[base + tag_idx] > 0.5:
                d = float(obs[base + DIST_IDX])
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _min_ball_dist(self, obs: np.ndarray) -> float:
        return self._min_tag_dist(obs, BALL_TAG_IDX)

    def _teammate_dist(self, obs: np.ndarray, agent_id: int) -> float:
        tag = BLUE_AGENT_TAG_IDX if agent_id < 2 else PURPLE_AGENT_TAG_IDX
        return self._min_tag_dist(obs, tag)

    def _shape(self, obs: dict, base_rewards: dict) -> dict:
        ball_dists = {aid: self._min_ball_dist(o) for aid, o in obs.items()}

        team_ids = [
            [aid for aid in ball_dists if aid < 2],
            [aid for aid in ball_dists if aid >= 2],
        ]
        closest = {}
        for team in team_ids:
            if team:
                closest[min(team, key=lambda a: ball_dists[a])] = True

        shaped = {}
        for aid, agent_obs in obs.items():
            dist = ball_dists[aid]
            bonus = 0.0

            bonus += self.ball_proximity_weight * max(0.0, 1.0 - dist)

            prev = self._prev_ball_dist.get(aid, 1.0)
            bonus += self.ball_progress_weight * (prev - dist)
            self._prev_ball_dist[aid] = dist

            if aid in closest:
                bonus += self.possession_weight

            if prev < KICK_THRESHOLD and (dist - prev) > KICK_ESCAPE_MIN:
                bonus += self.kick_weight

            teammate_dist = self._teammate_dist(agent_obs, aid)
            if teammate_dist > SPREAD_THRESHOLD:
                bonus += self.spread_weight

            shaped[aid] = float(base_rewards.get(aid, 0.0)) + bonus

        return shaped

    def reset(self, **kwargs):
        self._prev_ball_dist = {}
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        if isinstance(rewards, dict) and isinstance(obs, dict):
            rewards = self._shape(obs, rewards)
        return obs, rewards, dones, infos
