import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class MAPPOCentralCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO-style model using local observations for the actor and
    concatenated global state for the critic.

    Expected observation format per agent:
    {
        "obs": <local compact observation>,
        "state": <global state concatenating all agents and ball state>
    }
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        nn.Module.__init__(self)

        original_space = getattr(obs_space, "original_space", None)
        if original_space is None:
            original_space = obs_space
        if not hasattr(original_space, "spaces"):
            raise ValueError(
                "MAPPOCentralCriticModel requires gym.spaces.Dict observations"
            )
        if "obs" not in original_space.spaces or "state" not in original_space.spaces:
            raise ValueError(
                "MAPPOCentralCriticModel expects a Dict observation with keys 'obs' and 'state'"
            )

        local_obs_space = original_space.spaces["obs"]
        global_state_space = original_space.spaces["state"]

        critic_hiddens = model_config.get("custom_model_config", {}).get(
            "critic_hiddens",
            model_config.get("fcnet_hiddens", [256, 256]),
        )
        critic_model_config = dict(model_config)
        critic_model_config["fcnet_hiddens"] = critic_hiddens

        self.actor_model = FullyConnectedNetwork(
            local_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )
        self.critic_model = FullyConnectedNetwork(
            global_state_space,
            action_space,
            1,
            critic_model_config,
            name + "_critic",
        )

        self._last_global_state = None

    def forward(self, input_dict, state, seq_lens):
        local_obs = input_dict["obs"]["obs"].float()
        self._last_global_state = input_dict["obs"]["state"].float()
        logits, model_state = self.actor_model({"obs": local_obs}, state, seq_lens)
        return logits, model_state

    def value_function(self):
        if self._last_global_state is None:
            raise ValueError("value_function called before forward pass")
        values, _ = self.critic_model({"obs": self._last_global_state}, [], None)
        return torch.reshape(values, [-1])
