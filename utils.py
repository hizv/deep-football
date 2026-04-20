from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class CompactObservationRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    Compact multi-agent observation wrapper with optional PBRS shaping.

    Observations are built from kinematic game state exposed in env infos.
    If info fields are unavailable, this wrapper falls back to a compact
    projection of the raw observations.
    """

    DEFAULT_AGENT_IDS = (0, 1, 2, 3)

    def __init__(self, env, config=None):
        super().__init__(env)
        config = config or {}

        self.agent_ids = tuple(config.get("agent_ids", self.DEFAULT_AGENT_IDS))
        self.return_dict_obs = bool(config.get("return_dict_obs", False))
        self.use_pbrs = bool(config.get("use_pbrs", False))

        self.pbrs_alpha = float(config.get("pbrs_alpha", 1.0))
        self.pbrs_beta = float(config.get("pbrs_beta", 0.3))
        self.pbrs_gamma = float(config.get("pbrs_gamma", 0.99))
        self.pbrs_scale = float(config.get("pbrs_scale", 1.0))

        self.position_scale = float(config.get("position_scale", 20.0))
        self.velocity_scale = float(config.get("velocity_scale", 10.0))
        self.distance_scale = float(config.get("distance_scale", 30.0))
        self.yaw_rate_scale = float(config.get("yaw_rate_scale", 180.0))
        self.prediction_dt = float(config.get("prediction_dt", 0.2))
        self.prediction_horizons = tuple(config.get("prediction_horizons", (1, 2, 3)))

        goal_x = float(config.get("goal_x", 16.0))
        self.left_goal = np.asarray([-goal_x, 0.0], dtype=np.float32)
        self.right_goal = np.asarray([goal_x, 0.0], dtype=np.float32)

        self.local_obs_dim = 42
        self.global_state_dim = self.local_obs_dim * len(self.agent_ids) + 4

        local_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.local_obs_dim,),
            dtype=np.float32,
        )
        if self.return_dict_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "obs": local_obs_space,
                    "state": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.global_state_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = local_obs_space

        self.action_space = env.action_space
        self._prev_potential = {agent_id: 0.0 for agent_id in self.agent_ids}

    def _coerce_obs_dict(self, observations):
        if isinstance(observations, dict):
            return observations
        if not self.agent_ids:
            raise ValueError("CompactObservationRewardWrapper requires known agent ids")
        return {self.agent_ids[0]: observations}

    @staticmethod
    def _to_xy(value):
        if value is None:
            return np.zeros(2, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
        if arr.size == 1:
            return np.asarray([arr[0], 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _extract_player_state(self, agent_info):
        player_info = {}
        if isinstance(agent_info, dict):
            player_info = agent_info.get("player_info", agent_info)

        pos = self._to_xy(player_info.get("position"))
        vel = self._to_xy(player_info.get("velocity"))

        yaw_rate = player_info.get("yaw_rate")
        if yaw_rate is None:
            angular_velocity = player_info.get("angular_velocity")
            if angular_velocity is not None:
                flat_angular = np.asarray(angular_velocity, dtype=np.float32).reshape(-1)
                yaw_rate = flat_angular[-1] if flat_angular.size > 0 else 0.0
            else:
                yaw_rate = player_info.get("rotation_y", 0.0)

        has_state = "position" in player_info or "velocity" in player_info
        return {
            "pos": pos,
            "vel": vel,
            "yaw_rate": float(yaw_rate),
            "valid": bool(has_state),
        }

    def _extract_ball_state(self, infos):
        if not isinstance(infos, dict):
            return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False

        for agent_info in infos.values():
            if not isinstance(agent_info, dict):
                continue
            for key in ("ball_info", "ball_state", "ball"):
                ball_info = agent_info.get(key)
                if not isinstance(ball_info, dict):
                    continue
                has_state = "position" in ball_info or "velocity" in ball_info
                if has_state:
                    return (
                        self._to_xy(ball_info.get("position")),
                        self._to_xy(ball_info.get("velocity")),
                        True,
                    )

        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False

    def _teammate_and_opponents(self, agent_id):
        half = max(1, len(self.agent_ids) // 2)
        team_a = self.agent_ids[:half]
        team_b = self.agent_ids[half:]

        if agent_id in team_a:
            team_ids = team_a
            opp_ids = list(team_b)
            own_goal = self.left_goal
            opp_goal = self.right_goal
        else:
            team_ids = team_b if team_b else team_a
            opp_ids = list(team_a)
            own_goal = self.right_goal
            opp_goal = self.left_goal

        teammate = next((idx for idx in team_ids if idx != agent_id), agent_id)
        if not opp_ids:
            opp_ids = [agent_id, agent_id]
        elif len(opp_ids) == 1:
            opp_ids = [opp_ids[0], opp_ids[0]]

        return teammate, opp_ids[:2], own_goal, opp_goal

    def _distance_feature(self, vec):
        return float(np.linalg.norm(vec) / self.distance_scale)

    def _speed_feature(self, vec):
        return float(np.linalg.norm(vec) / self.velocity_scale)

    def _fallback_local_obs(self, raw_agent_obs):
        flat_obs = np.asarray(raw_agent_obs, dtype=np.float32).reshape(-1)
        compact_obs = np.zeros(self.local_obs_dim, dtype=np.float32)
        used = min(self.local_obs_dim, flat_obs.size)
        if used > 0:
            compact_obs[:used] = flat_obs[:used]
        return compact_obs

    def _build_local_obs(self, agent_id, player_states, ball_pos, ball_vel):
        own_state = player_states.get(agent_id, None)
        if own_state is None:
            return np.zeros(self.local_obs_dim, dtype=np.float32)

        teammate_id, opponent_ids, own_goal, opp_goal = self._teammate_and_opponents(agent_id)

        teammate_state = player_states.get(teammate_id, own_state)
        opp1_state = player_states.get(opponent_ids[0], own_state)
        opp2_state = player_states.get(opponent_ids[1], own_state)

        own_pos = own_state["pos"]
        own_vel = own_state["vel"]
        own_yaw_rate = own_state["yaw_rate"]

        rel_ball = ball_pos - own_pos
        rel_teammate = teammate_state["pos"] - own_pos
        rel_opp1 = opp1_state["pos"] - own_pos
        rel_opp2 = opp2_state["pos"] - own_pos

        rel_teammate_vel = teammate_state["vel"] - own_vel
        rel_opp1_vel = opp1_state["vel"] - own_vel
        rel_opp2_vel = opp2_state["vel"] - own_vel

        vec_to_own_goal = own_goal - own_pos
        vec_to_opp_goal = opp_goal - own_pos

        features = [
            *(own_pos / self.position_scale),
            *(own_vel / self.velocity_scale),
            own_yaw_rate / self.yaw_rate_scale,
            *(rel_ball / self.position_scale),
            self._distance_feature(rel_ball),
            *(ball_vel / self.velocity_scale),
            *(rel_teammate / self.position_scale),
            *(rel_teammate_vel / self.velocity_scale),
            *(rel_opp1 / self.position_scale),
            *(rel_opp1_vel / self.velocity_scale),
            *(rel_opp2 / self.position_scale),
            *(rel_opp2_vel / self.velocity_scale),
            *(vec_to_own_goal / self.position_scale),
            *(vec_to_opp_goal / self.position_scale),
        ]

        for horizon in self.prediction_horizons:
            predicted_ball = ball_pos + (self.prediction_dt * float(horizon)) * ball_vel
            features.extend((predicted_ball - own_pos) / self.position_scale)

        features.extend(
            [
                self._distance_feature(rel_teammate),
                self._distance_feature(rel_opp1),
                self._distance_feature(rel_opp2),
                self._distance_feature(vec_to_own_goal),
                self._distance_feature(vec_to_opp_goal),
                self._speed_feature(ball_vel),
                self._speed_feature(own_vel),
                self._speed_feature(teammate_state["vel"]),
                self._speed_feature(opp1_state["vel"]),
                self._speed_feature(opp2_state["vel"]),
            ]
        )

        compact_obs = np.asarray(features, dtype=np.float32).reshape(-1)
        if compact_obs.size == self.local_obs_dim:
            return compact_obs

        resized_obs = np.zeros(self.local_obs_dim, dtype=np.float32)
        used = min(self.local_obs_dim, compact_obs.size)
        resized_obs[:used] = compact_obs[:used]
        return resized_obs

    def _build_compact_observations(self, observations, infos):
        observations = self._coerce_obs_dict(observations)
        agent_ids = list(observations.keys())

        player_states = {}
        has_player_state = False
        for agent_id in agent_ids:
            state = self._extract_player_state(infos.get(agent_id, {}) if isinstance(infos, dict) else {})
            has_player_state = has_player_state or state["valid"]
            player_states[agent_id] = state

        ball_pos, ball_vel, has_ball_state = self._extract_ball_state(infos)

        compact_obs = {}
        if has_player_state and has_ball_state:
            for agent_id in agent_ids:
                compact_obs[agent_id] = self._build_local_obs(
                    agent_id,
                    player_states,
                    ball_pos,
                    ball_vel,
                )
        else:
            for agent_id in agent_ids:
                compact_obs[agent_id] = self._fallback_local_obs(observations[agent_id])

        return compact_obs, player_states, ball_pos, ball_vel

    def _build_global_state(self, compact_obs, ball_pos, ball_vel):
        state_parts = []
        for agent_id in self.agent_ids:
            state_parts.append(compact_obs.get(agent_id, np.zeros(self.local_obs_dim, dtype=np.float32)))
        state_parts.append(ball_pos / self.position_scale)
        state_parts.append(ball_vel / self.velocity_scale)
        return np.concatenate(state_parts).astype(np.float32)

    def _format_observation(self, compact_obs, global_state):
        if not self.return_dict_obs:
            return compact_obs

        formatted_obs = {}
        for agent_id, local_obs in compact_obs.items():
            formatted_obs[agent_id] = {
                "obs": local_obs,
                "state": global_state,
            }
        return formatted_obs

    def _compute_potential(self, player_states, ball_pos):
        potential = {}
        for agent_id in self.agent_ids:
            state = player_states.get(agent_id, None)
            if state is None or not state.get("valid", False):
                potential[agent_id] = 0.0
                continue

            _, _, _, opp_goal = self._teammate_and_opponents(agent_id)
            dist_ball_goal = np.linalg.norm(ball_pos - opp_goal)
            dist_agent_ball = np.linalg.norm(state["pos"] - ball_pos)
            potential[agent_id] = (
                self.pbrs_alpha / (1.0 + dist_ball_goal)
                + self.pbrs_beta / (1.0 + dist_agent_ball)
            )
        return potential

    def _shape_rewards(self, rewards, player_states, ball_pos):
        if not self.use_pbrs or not isinstance(rewards, dict):
            return rewards

        current_potential = self._compute_potential(player_states, ball_pos)
        shaped_rewards = {}
        for agent_id, reward in rewards.items():
            if agent_id == "__all__":
                shaped_rewards[agent_id] = reward
                continue
            prev = self._prev_potential.get(agent_id, 0.0)
            curr = current_potential.get(agent_id, 0.0)
            shaped_rewards[agent_id] = float(reward) + self.pbrs_scale * (
                self.pbrs_gamma * curr - prev
            )

        self._prev_potential = current_potential
        return shaped_rewards

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        compact_obs, player_states, ball_pos, ball_vel = self._build_compact_observations(
            raw_obs,
            infos={},
        )

        global_state = self._build_global_state(compact_obs, ball_pos, ball_vel)
        self._prev_potential = self._compute_potential(player_states, ball_pos)
        return self._format_observation(compact_obs, global_state)

    def step(self, action):
        raw_obs, rewards, dones, infos = self.env.step(action)

        compact_obs, player_states, ball_pos, ball_vel = self._build_compact_observations(
            raw_obs,
            infos=infos,
        )
        global_state = self._build_global_state(compact_obs, ball_pos, ball_vel)
        shaped_obs = self._format_observation(compact_obs, global_state)
        shaped_rewards = self._shape_rewards(rewards, player_states, ball_pos)

        return shaped_obs, shaped_rewards, dones, infos


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    env_config = dict(env_config)
    use_compact_obs = bool(env_config.pop("use_compact_obs", False))
    wrapper_config = {
        "agent_ids": env_config.pop("agent_ids", CompactObservationRewardWrapper.DEFAULT_AGENT_IDS),
        "return_dict_obs": bool(env_config.pop("return_dict_obs", False)),
        "use_pbrs": bool(env_config.pop("use_pbrs", False)),
        "pbrs_alpha": float(env_config.pop("pbrs_alpha", 1.0)),
        "pbrs_beta": float(env_config.pop("pbrs_beta", 0.3)),
        "pbrs_gamma": float(env_config.pop("pbrs_gamma", 0.99)),
        "pbrs_scale": float(env_config.pop("pbrs_scale", 1.0)),
        "position_scale": float(env_config.pop("position_scale", 20.0)),
        "velocity_scale": float(env_config.pop("velocity_scale", 10.0)),
        "distance_scale": float(env_config.pop("distance_scale", 30.0)),
        "yaw_rate_scale": float(env_config.pop("yaw_rate_scale", 180.0)),
        "prediction_dt": float(env_config.pop("prediction_dt", 0.2)),
        "prediction_horizons": tuple(env_config.pop("prediction_horizons", (1, 2, 3))),
        "goal_x": float(env_config.pop("goal_x", 16.0)),
    }

    env = soccer_twos.make(**env_config)

    if use_compact_obs or wrapper_config["use_pbrs"]:
        return CompactObservationRewardWrapper(env, wrapper_config)

    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
