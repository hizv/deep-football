"""Compatibility patches for soccer_twos/ml-agents under local training setups."""

import numpy as np
from mlagents_envs import environment as _mla_env


_TIMEOUT_PATCH_ATTR = "_deep_football_timeout_patched"


def apply_unity_compat(timeout_wait: int = 300) -> None:
    """Patch NumPy and Unity defaults expected by the bundled soccer stack."""
    numpy_aliases = {
        "bool": np.bool_,
        "object": object,
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
        "long": int,
    }
    for alias, target in numpy_aliases.items():
        if alias not in np.__dict__:
            setattr(np, alias, target)

    unity_env_init = _mla_env.UnityEnvironment.__init__
    if getattr(unity_env_init, _TIMEOUT_PATCH_ATTR, False):
        return

    def _patched_unity_env_init(self, *args, timeout_wait=timeout_wait, **kwargs):
        return unity_env_init(self, *args, timeout_wait=timeout_wait, **kwargs)

    setattr(_patched_unity_env_init, _TIMEOUT_PATCH_ATTR, True)
    _mla_env.UnityEnvironment.__init__ = _patched_unity_env_init
