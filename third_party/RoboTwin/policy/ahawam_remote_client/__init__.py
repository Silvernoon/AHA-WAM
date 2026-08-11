from typing import Any, Dict, Optional

import numpy as np


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    raise RuntimeError("The remote client does not construct a local policy model.")


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    if obs is None:
        raise ValueError("Remote policy evaluation requires an observation.")
    payload = dict(obs)
    payload["_instruction"] = TASK_ENV.get_instruction()
    action = model.call(func_name="predict_action", obs=payload)
    TASK_ENV.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
