"""Module for mlflow utilities"""

import os

from axolotl.utils.dict import DictDefault


def setup_mlflow_env_vars(cfg: DictDefault):
    for key in cfg.keys():
        if key.startswith("mlflow_"):
            value = cfg.get(key, "")

            if value and isinstance(value, str) and len(value) > 0:
                os.environ[key.upper()] = value

    # Enable mlflow if tracking uri is present
    if cfg.mlflow_tracking_uri and len(cfg.mlflow_tracking_uri) > 0:
        cfg.use_mlflow = True
        os.environ.pop("MLFLOW_TRACKING_DISABLED", None)  # Remove if present
    else:
        os.environ["MLFLOW_TRACKING_DISABLED"] = "true"
